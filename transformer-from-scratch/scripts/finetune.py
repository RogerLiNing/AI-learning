import os
import json
import torch
import logging
import argparse
import datetime
import numpy as np
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.transformer import TransformerLM, create_look_ahead_mask
from data.tokenizer import Tokenizer, BPETokenizer
from data.dataset import ClassificationDataset, create_dataloader


def setup_logging(log_dir):
    """设置日志"""
    os.makedirs(log_dir, exist_ok=True)
    
    # 获取当前时间
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"finetune_{timestamp}.log")
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    创建带有预热的线性学习率调度器
    
    参数:
        optimizer: 优化器
        num_warmup_steps: 预热步数
        num_training_steps: 总训练步数
        last_epoch: 上一轮的epoch
        
    返回:
        学习率调度器
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config


class TransformerForClassification(torch.nn.Module):
    """
    用于分类任务的Transformer模型
    """
    def __init__(self, transformer, num_classes, dropout=0.1):
        """
        初始化分类模型
        
        参数:
            transformer: 预训练的Transformer模型
            num_classes: 分类类别数
            dropout: dropout概率
        """
        super().__init__()
        
        # 使用预训练的Transformer模型
        self.transformer = transformer
        
        # 移除TransformerLM的最终输出层，我们不需要它进行分类
        self.d_model = 128  # 与我们的小型模型保持一致
        
        # 分类头
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(self.d_model, num_classes)
        
        # 初始化分类头参数
        self._init_weights()
        
    def _init_weights(self):
        """
        初始化分类头参数
        """
        self.classifier.weight.data.normal_(mean=0.0, std=0.02)
        self.classifier.bias.data.zero_()
        
    def _freeze_layers(self, num_layers_to_freeze=1):
        """
        冻结Transformer模型的某些层
        
        参数:
            num_layers_to_freeze: 要冻结的层数
        """
        # 冻结嵌入层
        for param in self.transformer.embedding.parameters():
            param.requires_grad = False
            
        # 冻结指定数量的解码器层
        for i in range(min(num_layers_to_freeze, len(self.transformer.decoder.layers))):
            for param in self.transformer.decoder.layers[i].parameters():
                param.requires_grad = False
                
    def forward(self, input_ids, attention_mask):
        """
        前向传播
        
        参数:
            input_ids: 输入ID [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            
        返回:
            logits: [batch_size, num_classes]
        """
        # 不直接使用transformer的forward方法，而是获取中间表示
        # 嵌入层
        x = self.transformer.embedding(input_ids)
        
        # 注意力掩码
        if attention_mask is not None:
            # 创建注意力掩码
            seq_len = input_ids.size(1)
            # 将填充掩码转换为注意力掩码
            # attention_mask: [batch_size, seq_len] -> [batch_size, 1, seq_len, seq_len]
            attn_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn_mask = attn_mask.expand(-1, -1, seq_len, -1)
        else:
            # 如果没有提供掩码，自动创建
            seq_len = input_ids.size(1)
            attn_mask = create_look_ahead_mask(seq_len).to(input_ids.device)
            attn_mask = attn_mask.unsqueeze(0).expand(input_ids.size(0), -1, -1)
        
        # 解码器层 - 直接调用获取隐藏状态
        x, _ = self.transformer.decoder(x, attn_mask)
        
        # 对序列进行平均池化，得到一个句子级别的表示
        # 首先确保attention_mask的形状为 [batch_size, seq_len]
        if attention_mask is None:
            # 如果没有提供掩码，则所有位置都有效
            attention_mask = torch.ones_like(input_ids)
            
        # 扩展掩码维度以便于广播 [batch_size, seq_len] -> [batch_size, seq_len, 1]
        mask_expanded = attention_mask.unsqueeze(-1).float()
        
        # 对有效token进行池化 [batch_size, seq_len, d_model]
        sum_embeddings = torch.sum(x * mask_expanded, dim=1)
        sum_mask = torch.sum(mask_expanded, dim=1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)  # 防止除以0
        
        # 计算平均值 [batch_size, d_model]
        sequence_output = sum_embeddings / sum_mask
        
        # 应用分类头
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        return logits


def load_pretrained_model(model_path, tokenizer):
    """
    加载预训练模型
    
    参数:
        model_path: 模型路径
        tokenizer: 分词器实例
        
    返回:
        预训练模型
    """
    # 加载检查点
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # 创建模型 - 使用与小型预训练模型相同的参数
    model = TransformerLM(
        vocab_size=len(tokenizer.token2id),
        d_model=128,  # 与我们的小型预训练模型匹配
        n_layers=2,   # 与我们的小型预训练模型匹配
        n_heads=2,    # 与我们的小型预训练模型匹配
        d_ff=512,     # 与我们的小型预训练模型匹配
        max_seq_len=128  # 与我们的小型预训练模型匹配
    )
    
    # 加载模型参数
    model.load_state_dict(checkpoint['model_state_dict'])
    
    logging.info(f"从 {model_path} 加载预训练模型")
    return model


def train(config, model, train_loader, val_loader=None, device='cuda'):
    """
    训练模型
    
    参数:
        config: 配置字典
        model: 模型实例
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器（可选）
        device: 训练设备
    """
    # 将模型移至指定设备
    model.to(device)
    
    # 设置优化器
    # 可以对不同层设置不同的学习率
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': config.get('weight_decay', 0.01)
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=config.get('learning_rate', 5e-5),
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # 训练参数
    num_epochs = config.get('num_epochs', 10)
    gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
    max_grad_norm = config.get('max_grad_norm', 1.0)
    
    # 学习率预热
    warmup_ratio = config.get('warmup_ratio', 0.1)
    total_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
    warmup_steps = int(total_steps * warmup_ratio)
    
    # 创建学习率调度器
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # 保存路径
    save_dir = config.get('save_dir', './saved_models')
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存频率
    save_steps = config.get('save_steps', 500)
    eval_steps = config.get('eval_steps', 100)
    
    # 日志参数
    logging_steps = config.get('logging_steps', 50)
    
    # 训练循环
    global_step = 0
    best_val_accuracy = 0.0
    
    logging.info("开始微调...")
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        # 训练一个epoch
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for step, batch in enumerate(train_iter):
            # 将数据移至设备
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # 前向传播
            logits = model(input_ids, attention_mask)
            
            # 计算损失
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            
            # 处理梯度累积
            loss = loss / gradient_accumulation_steps
            loss.backward()
            
            epoch_loss += loss.item() * gradient_accumulation_steps
            
            # 更新参数（每accumulation_steps步更新一次）
            if (step + 1) % gradient_accumulation_steps == 0:
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                # 更新参数
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                global_step += 1
                
                # 日志记录
                if global_step % logging_steps == 0:
                    lr = scheduler.get_last_lr()[0]
                    logging.info(f"Epoch: {epoch+1}/{num_epochs}, Step: {global_step}, Loss: {loss.item() * gradient_accumulation_steps:.4f}, LR: {lr:.8f}")
                
                # 保存模型
                if global_step % save_steps == 0:
                    save_path = os.path.join(save_dir, f"model_step_{global_step}.pt")
                    torch.save({
                        'epoch': epoch,
                        'global_step': global_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss': loss.item()
                    }, save_path)
                    logging.info(f"模型已保存至: {save_path}")
                
                # 评估模型
                if val_loader is not None and global_step % eval_steps == 0:
                    accuracy = evaluate(model, val_loader, device)
                    logging.info(f"验证准确率: {accuracy:.4f}")
                    
                    # 保存最佳模型
                    if accuracy > best_val_accuracy:
                        best_val_accuracy = accuracy
                        save_path = os.path.join(save_dir, "model_best.pt")
                        torch.save({
                            'epoch': epoch,
                            'global_step': global_step,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'accuracy': accuracy
                        }, save_path)
                        logging.info(f"最佳模型已保存至: {save_path} (准确率: {accuracy:.4f})")
                    
                    # 恢复训练模式
                    model.train()
        
        # 计算整个epoch的平均损失
        avg_loss = epoch_loss / len(train_loader)
        logging.info(f"Epoch {epoch+1}/{num_epochs} 完成，平均损失: {avg_loss:.4f}")
        
        # 每个epoch结束保存一次模型
        save_path = os.path.join(save_dir, f"model_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_loss
        }, save_path)
        logging.info(f"Epoch {epoch+1} 模型已保存至: {save_path}")
        
        # 在每个epoch结束时评估一次
        if val_loader is not None:
            accuracy = evaluate(model, val_loader, device)
            logging.info(f"Epoch {epoch+1} 验证准确率: {accuracy:.4f}")
    
    # 保存最终模型
    save_path = os.path.join(save_dir, "model_final.pt")
    torch.save({
        'epoch': num_epochs,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': avg_loss
    }, save_path)
    logging.info(f"最终模型已保存至: {save_path}")


def evaluate(model, data_loader, device):
    """
    评估分类模型
    
    参数:
        model: 模型实例
        data_loader: 数据加载器
        device: 设备
        
    返回:
        准确率
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="评估"):
            # 将数据移至设备
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # 前向传播
            logits = model(input_ids, attention_mask)
            
            # 计算准确率
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    # 计算准确率
    accuracy = correct / total
    return accuracy


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Transformer 微调脚本")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--pretrained_model", type=str, required=True, help="预训练模型路径")
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置日志
    log_dir = config.get('log_dir', './logs')
    setup_logging(log_dir)
    
    # 记录配置
    logging.info(f"配置: {json.dumps(config, indent=2, ensure_ascii=False)}")
    logging.info(f"预训练模型路径: {args.pretrained_model}")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"使用设备: {device}")
    
    # 加载分词器
    tokenizer_path = config.get('tokenizer_path')
    if not tokenizer_path or not os.path.exists(tokenizer_path):
        raise ValueError(f"分词器路径不存在: {tokenizer_path}")
    
    # 根据分词器类型加载
    if config.get('tokenizer_type', 'basic') == 'bpe':
        tokenizer = BPETokenizer.from_file(tokenizer_path)
    else:
        tokenizer = Tokenizer.from_file(tokenizer_path)
    
    logging.info(f"从 {tokenizer_path} 加载分词器")
    
    # 加载训练数据
    train_data_path = config.get('train_data_path')
    val_data_path = config.get('val_data_path')
    
    if not train_data_path or not os.path.exists(train_data_path):
        raise ValueError(f"训练数据路径不存在: {train_data_path}")
    
    # 创建数据集
    max_seq_len = config.get('max_seq_len', 512)
    
    # 读取类别数量
    num_classes = config.get('num_classes')
    if num_classes is None:
        raise ValueError("配置中必须指定类别数量 (num_classes)")
    
    # 加载训练集
    train_dataset = ClassificationDataset.from_file(
        train_data_path,
        tokenizer=tokenizer,
        max_length=max_seq_len,
        text_field=config.get('text_field', 'text'),
        label_field=config.get('label_field', 'label')
    )
    
    # 加载验证集（如果存在）
    val_dataset = None
    if val_data_path and os.path.exists(val_data_path):
        val_dataset = ClassificationDataset.from_file(
            val_data_path,
            tokenizer=tokenizer,
            max_length=max_seq_len,
            text_field=config.get('text_field', 'text'),
            label_field=config.get('label_field', 'label')
        )
    
    # 创建数据加载器
    batch_size = config.get('batch_size', 16)
    
    train_loader = create_dataloader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=config.get('num_workers', 4)
    )
    
    val_loader = None
    if val_dataset:
        val_loader = create_dataloader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=config.get('num_workers', 4)
        )
    
    # 加载预训练模型
    pretrained_model = load_pretrained_model(args.pretrained_model, tokenizer)
    
    # 创建分类模型
    model = TransformerForClassification(
        transformer=pretrained_model,
        num_classes=num_classes,
        dropout=config.get('dropout', 0.1)
    )
    
    logging.info(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    logging.info(f"可训练参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 训练模型
    train(config, model, train_loader, val_loader, device)


if __name__ == "__main__":
    main()
