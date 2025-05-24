#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用HuggingFace分词器和对话数据预训练Transformer模型
"""

import os
import sys
import json
import torch
import logging
import argparse
import platform
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

# 混合精度训练
import contextlib
from torch.amp import autocast, GradScaler

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.transformer import TransformerLM
from data.hf_tokenizer import HFTokenizer
from data.conversation_dataset import ConversationDataset
from torch.utils.data import DataLoader

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config

def setup_logging(log_dir):
    """设置日志"""
    os.makedirs(log_dir, exist_ok=True)
    
    # 添加文件处理器
    file_handler = logging.FileHandler(os.path.join(log_dir, 'train.log'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    # 添加到根日志记录器
    logging.getLogger().addHandler(file_handler)

def create_model(config, tokenizer):
    """创建模型"""
    model = TransformerLM(
        vocab_size=tokenizer.vocab_size,
        d_model=config.get('d_model', 512),
        n_layers=config.get('n_layers', 6),
        n_heads=config.get('n_heads', 8),
        d_ff=config.get('d_ff', 2048),
        dropout=config.get('dropout', 0.1),
        max_seq_len=config.get('max_seq_len', 512)
    )
    
    logging.info(f"创建模型: d_model={config.get('d_model')}, n_layers={config.get('n_layers')}, n_heads={config.get('n_heads')}")
    logging.info(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    return model

def create_dataloader(dataset, batch_size, shuffle=True, num_workers=4):
    """创建数据加载器"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    创建带有预热的线性学习率调度器
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)

def train(config, model, train_loader, val_loader=None, device='cuda', use_amp=True):
    """训练模型"""
    # 模型和数据诊断信息
    logging.info(f"\n=== 模型与数据诊断信息 ===")
    
    # 获取词表大小 - 从模型的嵌入层获取
    embedding_layer = model.embedding.token_embedding.embedding
    vocab_size = embedding_layer.num_embeddings
    logging.info(f"\u6a21型词表大小: {vocab_size}")
    
    # 检查第一个批次的数据
    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch['input_ids']
        max_id = input_ids.max().item()
        min_id = input_ids.min().item()
        unique_ids = torch.unique(input_ids)
        logging.info(f"\u6570据集中的token ID范围: {min_id} 到 {max_id}")
        logging.info(f"\u6570据集中不同的token ID数量: {len(unique_ids)}")
        
        if max_id >= vocab_size:
            logging.warning(f"\u8b66告: 数据集中有token ID超出词表范围! \u6700大ID {max_id} >= 词表大小 {vocab_size}")
        break
    
    logging.info(f"=== 诊断信息结束 ===\n")
    
    # 优化器和学习率调度器
    optimizer = AdamW(
        model.parameters(),
        lr=config.get('learning_rate', 5e-5),
        weight_decay=config.get('weight_decay', 0.01)
    )
    
    # 计算总训练步数
    num_epochs = config.get('num_epochs', 10)
    num_training_steps = len(train_loader) * num_epochs
    num_warmup_steps = int(num_training_steps * config.get('warmup_ratio', 0.1))
    
    # 创建学习率调度器
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # 损失函数
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)  # 忽略PAD标记(ID=0)
    
    # 混合精度训练设置
    if use_amp and device != 'cpu':
        device_type = 'cuda' if device.startswith('cuda') else device
        # 使用不带device_type参数的GradScaler (适用于PyTorch较早版本)
        scaler = GradScaler()
        # 但autocast仍然使用device_type参数
        autocast_fn = lambda: autocast(device_type=device_type)
    else:
        autocast_fn = contextlib.nullcontext
        scaler = None
    
    # 训练设置
    gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
    logging_steps = config.get('logging_steps', 100)
    save_steps = config.get('save_steps', 1000)
    eval_steps = config.get('eval_steps', 500)
    
    # 保存目录
    save_dir = config.get('save_dir', './saved_models')
    os.makedirs(save_dir, exist_ok=True)
    
    # 训练状态
    global_step = 0
    train_losses = []
    
    logging.info(f"开始训练... 设备: {device}, 混合精度: {use_amp if device != 'cpu' else False}")
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0
        epoch_steps = 0
        
        # 创建进度条
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
        
        for step, batch in enumerate(progress_bar):
            # 将数据移动到设备
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 混合精度训练
            if use_amp and device != 'cpu':
                # 获取正确的设备类型
                device_type = 'cuda' if device.startswith('cuda') else device
                with autocast(device_type=device_type):
                    # 前向传播
                    outputs = model(input_ids, attention_mask)
                    
                    # 计算损失
                    logits = outputs.view(-1, outputs.size(-1))
                    labels = labels.view(-1)
                    loss = loss_fn(logits, labels)
                    
                    # 如果使用梯度累积，需要除以累积步数
                    if gradient_accumulation_steps > 1:
                        loss = loss / gradient_accumulation_steps
                
                # 使用scaler进行反向传播
                scaler.scale(loss).backward()
            else:
                # 前向传播 (不使用混合精度)
                outputs = model(input_ids, attention_mask)
                
                # 计算损失
                logits = outputs.view(-1, outputs.size(-1))
                labels = labels.view(-1)
                loss = loss_fn(logits, labels)
                
                # 如果使用梯度累积，需要除以累积步数
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps
                    
                # 标准反向传播
                loss.backward()
            
            # 更新参数（每 gradient_accumulation_steps 步）
            if (step + 1) % gradient_accumulation_steps == 0:
                if use_amp and device != 'cpu':
                    # 混合精度更新
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # 标准更新
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
                    optimizer.step()
                
                scheduler.step()
                optimizer.zero_grad()
                
                # 更新全局步数
                global_step += 1
                
                # 记录损失
                epoch_loss += loss.item() * gradient_accumulation_steps
                epoch_steps += 1
                
                # 更新进度条
                progress_bar.set_postfix({'loss': loss.item() * gradient_accumulation_steps})
                
                # 每 logging_steps 步记录一次损失
                if global_step % logging_steps == 0:
                    avg_loss = epoch_loss / epoch_steps
                    logging.info(f"Epoch {epoch}, Step {global_step}: loss = {avg_loss:.4f}, lr = {scheduler.get_last_lr()[0]:.7f}")
                
                # 每 eval_steps 步评估一次模型
                if val_loader and global_step % eval_steps == 0:
                    val_loss = evaluate(model, val_loader, device, use_amp)
                    logging.info(f"Validation loss: {val_loss:.4f}")
                    model.train()  # 切回训练模式
                
                # 每 save_steps 步保存一次模型
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
                    logging.info(f"Model saved at step {global_step} to {save_path}")

                # 释放缓存
                if device == 'cuda':
                    torch.cuda.empty_cache()
        
        # 计算并记录epoch平均损失
        avg_epoch_loss = epoch_loss / epoch_steps
        train_losses.append(avg_epoch_loss)
        logging.info(f"Epoch {epoch} 完成. 平均损失: {avg_epoch_loss:.4f}")
        
        # 每个epoch结束后评估一次模型
        if val_loader:
            val_loss = evaluate(model, val_loader, device)
            logging.info(f"Epoch {epoch} 验证损失: {val_loss:.4f}")
        
        # 每个epoch保存一次模型
        save_path = os.path.join(save_dir, f"model_epoch_{epoch}.pt")
        torch.save({
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_epoch_loss
        }, save_path)
        logging.info(f"Epoch {epoch} 模型已保存至: {save_path}")
    
    # 保存最终模型
    final_path = os.path.join(save_dir, "model_final.pt")
    torch.save({
        'epoch': num_epochs,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'loss': train_losses[-1] if train_losses else 0.0
    }, final_path)
    logging.info(f"最终模型已保存至: {final_path}")
    
    return model

def evaluate(model, val_loader, device='cuda', use_amp=True):
    """评估模型"""
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)  # 忽略PAD标记
    total_loss = 0
    total_steps = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            # 将数据移动到设备
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 混合精度前向传播
            if use_amp and device != 'cpu':
                # 获取正确的设备类型
                device_type = 'cuda' if device.startswith('cuda') else device
                with autocast(device_type=device_type):
                    # 前向传播
                    outputs = model(input_ids, attention_mask)
                    
                    # 计算损失
                    logits = outputs.view(-1, outputs.size(-1))
                    labels = labels.view(-1)
                    loss = loss_fn(logits, labels)
            else:
                # 前向传播
                outputs = model(input_ids, attention_mask)
                
                # 计算损失
                logits = outputs.view(-1, outputs.size(-1))
                labels = labels.view(-1)
                loss = loss_fn(logits, labels)
            
            total_loss += loss.item()
            total_steps += 1
            
            # 释放缓存
            if device == 'cuda':
                torch.cuda.empty_cache()
    
    avg_loss = total_loss / total_steps
    return avg_loss

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="使用HuggingFace分词器和对话数据预训练Transformer模型")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--pretrained_model_name", type=str, default="bert-base-chinese", 
                        help="预训练模型名称，用于初始化分词器")
    parser.add_argument("--no_cuda", action="store_true", help="强制使用CPU训练")
    parser.add_argument("--no_amp", action="store_true", help="不使用混合精度训练")
    parser.add_argument("--no_mps", action="store_true", help="不使用Apple MPS加速")
    parser.add_argument("--gpu_id", type=int, default=0, help="要使用的GPU ID，多个GPU时有效")
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置日志
    log_dir = config.get('log_dir', './logs')
    setup_logging(log_dir)
    
    # 记录配置
    logging.info(f"配置: {json.dumps(config, indent=2, ensure_ascii=False)}")
    
    # 检测设备
    
    # 检测可用的计算设备
    use_cuda = torch.cuda.is_available() and not args.no_cuda
    use_mps = (not args.no_mps) and platform.processor() == 'arm' and torch.backends.mps.is_available()
    
    # 设置设备
    if use_cuda:
        if torch.cuda.device_count() > 1:
            logging.info(f"发现{torch.cuda.device_count()}个GPU设备, 使用GPU {args.gpu_id}")
            torch.cuda.set_device(args.gpu_id)
        device = f"cuda:{args.gpu_id}" if torch.cuda.device_count() > 1 else "cuda"
        
        # 记录CUDA GPU信息
        logging.info(f"使用设备: {device} - {torch.cuda.get_device_name(args.gpu_id if torch.cuda.device_count() > 1 else 0)}")
        logging.info(f"CUDA版本: {torch.version.cuda}")
        logging.info(f"cuDNN版本: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'}")
        logging.info(f"GPU显存: {torch.cuda.get_device_properties(args.gpu_id if torch.cuda.device_count() > 1 else 0).total_memory / 1024 / 1024 / 1024:.2f} GB")
        
        # 设置cuDNN加速
        torch.backends.cudnn.benchmark = True
        logging.info("cuDNN benchmark 模式已启用")
    elif use_mps:
        # 使用Apple Silicon MPS加速
        device = "mps"
        logging.info(f"使用设备: {device} (Apple Silicon GPU)")
        logging.info(f"Mac处理器: {platform.processor()}, PyTorch版本: {torch.__version__}")
    else:
        device = "cpu"
        logging.info(f"使用设备: CPU")
    
    # 设置混合精度训练
    use_amp = use_cuda and not args.no_amp  # MPS目前不支持混合精度
    if use_amp:
        logging.info("启用混合精度训练 (AMP)")
    elif use_cuda:
        logging.info("混合精度训练已禁用")
    
    
    # 初始化HuggingFace分词器
    tokenizer = HFTokenizer(
        pretrained_model_name=args.pretrained_model_name,
        add_special_tokens=True
    )
    logging.info(f"初始化HuggingFace分词器: {args.pretrained_model_name}, 词表大小: {tokenizer.vocab_size}")
    
    # 保存分词器
    tokenizer_dir = config.get('tokenizer_dir', 'data/hf_tokenizer')
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer.save(tokenizer_dir)
    logging.info(f"分词器已保存至: {tokenizer_dir}")
    
    # 加载训练数据
    train_data_path = config.get('train_data_path')
    val_data_path = config.get('val_data_path')
    
    if not train_data_path or not os.path.exists(train_data_path):
        raise ValueError(f"训练数据路径不存在: {train_data_path}")
    
    # 创建数据集
    max_seq_len = config.get('max_seq_len', 512)
    batch_size = config.get('batch_size', 16)
    
    # 加载训练集
    train_dataset = ConversationDataset.from_file(
        train_data_path,
        tokenizer=tokenizer,
        max_length=max_seq_len,
        add_special_tokens=False  # 数据中已经包含特殊标记
    )
    
    # 加载验证集（如果存在）
    val_dataset = None
    if val_data_path and os.path.exists(val_data_path):
        val_dataset = ConversationDataset.from_file(
            val_data_path,
            tokenizer=tokenizer,
            max_length=max_seq_len,
            add_special_tokens=False  # 数据中已经包含特殊标记
        )
    
    # 创建数据加载器
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
    
    # 创建模型
    model = create_model(config, tokenizer)
    model.to(device)
    
    # 输出模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"模型总参数: {total_params:,} (可训练: {trainable_params:,})")
    
    # 训练模型
    train(config, model, train_loader, val_loader, device=device, use_amp=use_amp)
    
    logging.info("预训练完成！")

if __name__ == "__main__":
    main()
