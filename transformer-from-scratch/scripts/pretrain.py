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

from model.transformer import TransformerLM
from data.tokenizer import Tokenizer, BPETokenizer
from data.dataset import MLMDataset, CLMDataset, create_dataloader


def setup_logging(log_dir):
    """设置日志"""
    os.makedirs(log_dir, exist_ok=True)
    
    # 获取当前时间
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"pretrain_{timestamp}.log")
    
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


def build_tokenizer(config, texts=None):
    """
    构建或加载分词器
    
    参数:
        config: 配置字典
        texts: 用于训练分词器的文本（如果需要）
        
    返回:
        分词器实例
    """
    # 检查是否存在预训练的分词器
    tokenizer_path = config.get('tokenizer_path')
    
    if tokenizer_path and os.path.exists(tokenizer_path):
        logging.info(f"加载现有分词器: {tokenizer_path}")
        if config.get('tokenizer_type', 'basic') == 'bpe':
            tokenizer = BPETokenizer.from_file(tokenizer_path)
        else:
            tokenizer = Tokenizer.from_file(tokenizer_path)
    else:
        # 创建新分词器
        vocab_size = config.get('vocab_size', 30000)
        logging.info(f"创建新分词器，词表大小: {vocab_size}")
        
        if config.get('tokenizer_type', 'basic') == 'bpe':
            tokenizer = BPETokenizer(vocab_size=vocab_size)
            
            if texts:
                # 训练BPE分词器
                num_merges = config.get('bpe_merges', 20000)
                logging.info(f"训练BPE分词器，合并操作数: {num_merges}")
                tokenizer.learn_bpe(texts, num_merges=num_merges)
        else:
            tokenizer = Tokenizer(vocab_size=vocab_size)
            
            if texts:
                # 构建基本词表
                logging.info("构建基本词表")
                tokenizer.build_vocab(texts, min_freq=config.get('min_word_freq', 5))
        
        # 保存分词器
        if tokenizer_path:
            os.makedirs(os.path.dirname(tokenizer_path), exist_ok=True)
            tokenizer.save(tokenizer_path)
            logging.info(f"分词器已保存至: {tokenizer_path}")
    
    return tokenizer


def load_training_data(config):
    """
    加载训练数据
    
    参数:
        config: 配置字典
        
    返回:
        训练文本列表
    """
    data_path = config.get('data_path')
    
    if not data_path or not os.path.exists(data_path):
        raise ValueError(f"数据路径不存在: {data_path}")
    
    logging.info(f"加载训练数据: {data_path}")
    
    # 读取数据
    texts = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="读取数据"):
            line = line.strip()
            if line:
                texts.append(line)
    
    logging.info(f"加载了 {len(texts)} 个训练样本")
    return texts


def create_model(config, vocab_size):
    """
    创建模型
    
    参数:
        config: 配置字典
        vocab_size: 词表大小
        
    返回:
        模型实例
    """
    model_type = config.get('model_type', 'lm')
    
    if model_type == 'lm':
        # 语言模型（仅解码器）
        model = TransformerLM(
            vocab_size=vocab_size,
            d_model=config.get('d_model', 512),
            n_layers=config.get('n_layers', 6),
            n_heads=config.get('n_heads', 8),
            d_ff=config.get('d_ff', 2048),
            max_seq_len=config.get('max_seq_len', 512),
            dropout=config.get('dropout', 0.1)
        )
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
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
    optimizer = AdamW(
        model.parameters(),
        lr=config.get('learning_rate', 5e-5),
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=config.get('weight_decay', 0.01)
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
    save_steps = config.get('save_steps', 10000)
    eval_steps = config.get('eval_steps', 5000)
    
    # 日志参数
    logging_steps = config.get('logging_steps', 100)
    
    # 训练循环
    global_step = 0
    best_val_loss = float('inf')
    
    logging.info("开始训练...")
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        # 训练一个epoch
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for step, batch in enumerate(train_iter):
            # 将数据移至设备
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 前向传播
            outputs = model(input_ids, attention_mask)
            
            # 计算损失
            # 调整形状以匹配交叉熵损失的要求
            logits = outputs.view(-1, outputs.size(-1))  # [batch_size*seq_len, vocab_size]
            labels = labels.view(-1)  # [batch_size*seq_len]
            
            # 计算交叉熵损失（忽略填充token，通常标记为-100）
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
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
                    val_loss = evaluate(model, val_loader, device)
                    logging.info(f"验证损失: {val_loss:.4f}")
                    
                    # 保存最佳模型
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_path = os.path.join(save_dir, "model_best.pt")
                        torch.save({
                            'epoch': epoch,
                            'global_step': global_step,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'loss': val_loss
                        }, save_path)
                        logging.info(f"最佳模型已保存至: {save_path}")
                    
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
    评估模型
    
    参数:
        model: 模型实例
        data_loader: 数据加载器
        device: 设备
        
    返回:
        验证损失
    """
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="评估"):
            # 将数据移至设备
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 前向传播
            outputs = model(input_ids, attention_mask)
            
            # 计算损失
            logits = outputs.view(-1, outputs.size(-1))
            labels = labels.view(-1)
            
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits, labels)
            
            total_loss += loss.item()
    
    # 计算平均损失
    avg_loss = total_loss / len(data_loader)
    return avg_loss


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Transformer 预训练脚本")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置日志
    log_dir = config.get('log_dir', './logs')
    setup_logging(log_dir)
    
    # 记录配置
    logging.info(f"配置: {json.dumps(config, indent=2, ensure_ascii=False)}")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"使用设备: {device}")
    
    # 加载训练数据
    texts = load_training_data(config)
    
    # 构建分词器
    tokenizer = build_tokenizer(config, texts)
    
    # 创建数据集
    pretraining_task = config.get('pretraining_task', 'mlm')
    max_seq_len = config.get('max_seq_len', 512)
    
    if pretraining_task == 'mlm':
        # 掩码语言模型
        mlm_probability = config.get('mlm_probability', 0.15)
        dataset = MLMDataset(
            texts=texts, 
            tokenizer=tokenizer, 
            max_length=max_seq_len, 
            mlm_probability=mlm_probability
        )
    elif pretraining_task == 'clm':
        # 因果语言模型
        dataset = CLMDataset(
            texts=texts, 
            tokenizer=tokenizer, 
            max_length=max_seq_len
        )
    else:
        raise ValueError(f"不支持的预训练任务: {pretraining_task}")
    
    # 分割数据集
    val_size = int(len(dataset) * config.get('val_ratio', 0.1))
    train_size = len(dataset) - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    logging.info(f"训练集大小: {train_size}, 验证集大小: {val_size}")
    
    # 创建数据加载器
    batch_size = config.get('batch_size', 8)
    
    train_loader = create_dataloader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=config.get('num_workers', 4)
    )
    
    val_loader = create_dataloader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=config.get('num_workers', 4)
    )
    
    # 创建模型
    model = create_model(config, len(tokenizer.token2id))
    logging.info(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练模型
    train(config, model, train_loader, val_loader, device)


if __name__ == "__main__":
    main()
