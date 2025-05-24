#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
微调预训练的Transformer模型用于问答任务
"""

import os
import sys
import json
import torch
import logging
import argparse
import time
from tqdm import tqdm
from typing import List, Dict, Any
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

# 混合精度训练
from torch.amp import autocast, GradScaler

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.transformer import TransformerLM
from data.hf_tokenizer import HFTokenizer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("finetune.log", mode='w')
    ]
)

def setup_device(use_cuda, use_mps):
    """设置设备"""
    if use_cuda and torch.cuda.is_available():
        device = "cuda"
        logging.info(f"使用设备: {device} - {torch.cuda.get_device_name()}")
    elif use_mps and torch.backends.mps.is_available():
        device = "mps"
        logging.info(f"使用设备: {device} - Apple Silicon")
    else:
        device = "cpu"
        logging.info(f"使用设备: {device}")
    
    return device

def load_jsonl(file_path, max_samples=None):
    """加载JSONL格式的数据"""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples is not None and i >= max_samples:
                    break
                if line.strip():
                    example = json.loads(line)
                    data.append(example)
        logging.info(f"从 {file_path} 加载了 {len(data)} 个问答对")
    except Exception as e:
        logging.error(f"加载 {file_path} 时出错: {e}")
    
    return data

class QADataset(Dataset):
    """问答数据集"""
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        question = item["question"]
        answer = item["answer"]
        
        # 构建输入文本: 问题 + 回答
        # 使用特殊标记来分隔问题和回答
        full_text = question + " [SEP] " + answer
        
        # 编码
        encodings = self.tokenizer.encode_plus(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # 创建标签
        # 输入部分不计算损失，回答部分计算损失
        question_ids = self.tokenizer.encode(question + " [SEP] ")
        question_len = len(question_ids)
        
        # 标签: 问题部分设为-100(忽略)，回答部分设为对应的token id
        labels = encodings["input_ids"].clone()
        labels[0, :question_len] = -100  # 忽略问题部分
        
        return {
            "input_ids": encodings["input_ids"].squeeze(),
            "attention_mask": encodings["attention_mask"].squeeze(),
            "labels": labels.squeeze()
        }

def safe_process_batch(batch, vocab_size, device):
    """安全处理批次数据，确保所有ID都在有效范围内"""
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    
    # 检查并限制输入ID范围
    max_input_id = input_ids.max().item()
    if max_input_id >= vocab_size:
        logging.warning(f"输入中发现超出范围的token ID: {max_input_id} >= {vocab_size}")
        logging.warning(f"自动将超出范围的ID限制在词表大小范围内")
        input_ids = torch.clamp(input_ids, 0, vocab_size-1)
    
    # 检查并限制标签ID范围
    # 跳过忽略的标签(-100)
    valid_labels = labels[labels != -100]
    if len(valid_labels) > 0:
        max_label_id = valid_labels.max().item()
        if max_label_id >= vocab_size:
            logging.warning(f"标签中发现超出范围的token ID: {max_label_id} >= {vocab_size}")
            logging.warning(f"自动将超出范围的标签限制在词表大小范围内")
            # 只对非-100的值进行约束
            labels = torch.where(
                labels != -100,
                torch.clamp(labels, 0, vocab_size-1),
                labels
            )
    
    return input_ids, attention_mask, labels

def load_tokenizer(tokenizer_path):
    """加载分词器"""
    if os.path.isdir(tokenizer_path):
        tokenizer = HFTokenizer(pretrained_model_name=tokenizer_path)
        logging.info(f"从本地加载分词器: {tokenizer_path}, 词表大小: {tokenizer.vocab_size}")
    else:
        logging.error(f"分词器路径不存在: {tokenizer_path}")
        raise FileNotFoundError(f"分词器路径不存在: {tokenizer_path}")
    return tokenizer

def load_model(model_path, config, device):
    """加载模型"""
    if not os.path.exists(model_path):
        logging.error(f"模型路径不存在: {model_path}")
        raise FileNotFoundError(f"模型路径不存在: {model_path}")
    
    # 加载检查点
    checkpoint = torch.load(model_path, map_location=device)
    
    # 从配置中获取模型参数
    d_model = config.get('d_model', 512)
    n_layers = config.get('n_layers', 6)
    n_heads = config.get('n_heads', 8)
    d_ff = config.get('d_ff', 2048)
    dropout = config.get('dropout', 0.1)
    max_seq_len = config.get('max_seq_len', 512)
    
    # 获取词表大小
    if 'vocab_size' in checkpoint:
        vocab_size = checkpoint['vocab_size']
    else:
        # 从分词器获取
        tokenizer = load_tokenizer(config.get('tokenizer_dir', 'data/hf_tokenizer'))
        vocab_size = tokenizer.vocab_size
    
    logging.info(f"创建模型: d_model={d_model}, n_layers={n_layers}, n_heads={n_heads}, vocab_size={vocab_size}")
    
    # 创建模型
    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        dropout=dropout,
        max_seq_len=max_seq_len
    )
    
    # 加载模型权重
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    
    return model

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """创建带有预热的线性学习率调度器"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)

def finetune(config, model, train_loader, val_loader=None, device='cuda', use_amp=True):
    """微调模型"""
    # 获取词表大小
    embedding_layer = model.embedding.token_embedding.embedding
    vocab_size = embedding_layer.num_embeddings
    logging.info(f"模型词表大小: {vocab_size}")
    
    # 损失函数 - 忽略-100标签
    num_epochs = config.get('num_epochs', 3)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    
    # 混合精度训练设置
    if use_amp and device != 'cpu':
        device_type = 'cuda' if device.startswith('cuda') else device
        scaler = GradScaler()
        autocast_fn = lambda: autocast(device_type=device_type)
    else:
        from contextlib import nullcontext
        autocast_fn = nullcontext
        scaler = None
    
    # 训练设置
    gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
    logging_steps = config.get('logging_steps', 10)
    save_steps = config.get('save_steps', 100)
    eval_steps = config.get('eval_steps', 50)
    
    # 保存目录
    save_dir = config.get('save_dir', './saved_models/finetuned')
    os.makedirs(save_dir, exist_ok=True)
    
    # 优化器 - 使用较小的学习率进行微调
    optimizer = AdamW(
        model.parameters(),
        lr=config.get('learning_rate', 1e-5),  # 微调使用较小的学习率
        weight_decay=config.get('weight_decay', 0.01)
    )
    
    # 学习率调度器
    total_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
    warmup_steps = int(total_steps * config.get('warmup_ratio', 0.1))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )
    
    logging.info(f"开始微调... 设备: {device}, 混合精度: {use_amp if device != 'cpu' else False}")
    logging.info(f"微调参数: 轮次={num_epochs}, 批次大小={config.get('batch_size', 8)}, 学习率={config.get('learning_rate', 1e-5)}")
    
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0
        epoch_steps = 0
        
        # 创建进度条
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
        
        for step, batch in enumerate(progress_bar):
            # 安全处理批次数据
            input_ids, attention_mask, labels = safe_process_batch(batch, vocab_size, device)
            
            # 混合精度训练
            if use_amp and device != 'cpu':
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
                
                # 学习率调度
                scheduler.step()
                optimizer.zero_grad()
                
                # 更新状态
                global_step += 1
                epoch_loss += loss.item() * gradient_accumulation_steps
                epoch_steps += 1
                
                # 更新进度条
                progress_bar.set_postfix(loss=epoch_loss / epoch_steps)
                
                # 记录日志
                if global_step % logging_steps == 0:
                    logging.info(f"Epoch {epoch}, Step {global_step}: loss = {loss.item() * gradient_accumulation_steps:.6f}")
                
                # 保存检查点
                if global_step % save_steps == 0:
                    checkpoint_path = os.path.join(save_dir, f"checkpoint-{global_step}")
                    os.makedirs(checkpoint_path, exist_ok=True)
                    
                    # 保存模型和训练状态
                    torch.save({
                        'epoch': epoch,
                        'global_step': global_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'scaler_state_dict': scaler.state_dict() if scaler else None,
                        'loss': epoch_loss / epoch_steps,
                        'vocab_size': vocab_size,
                        'd_model': config.get('d_model', 512),
                        'n_layers': config.get('n_layers', 6),
                        'n_heads': config.get('n_heads', 8),
                    }, os.path.join(checkpoint_path, "pytorch_model.bin"))
                    
                    logging.info(f"检查点已保存至 {checkpoint_path}")
                
                # 评估模型
                if val_loader and global_step % eval_steps == 0:
                    val_loss = evaluate(model, val_loader, loss_fn, device, use_amp)
                    logging.info(f"Step {global_step} | 验证损失: {val_loss:.6f}")
                    
                    # 保存最佳模型
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model_path = os.path.join(save_dir, "best_model")
                        os.makedirs(best_model_path, exist_ok=True)
                        torch.save({
                            'epoch': epoch,
                            'global_step': global_step,
                            'model_state_dict': model.state_dict(),
                            'loss': val_loss,
                            'vocab_size': vocab_size,
                            'd_model': config.get('d_model', 512),
                            'n_layers': config.get('n_layers', 6),
                            'n_heads': config.get('n_heads', 8),
                        }, os.path.join(best_model_path, "pytorch_model.bin"))
                        logging.info(f"新的最佳模型已保存至 {best_model_path}")
                    
                    model.train()  # 切回训练模式
                
                # 释放GPU缓存
                if device == 'cuda':
                    torch.cuda.empty_cache()
        
        # 每个epoch结束后的平均损失
        epoch_loss = epoch_loss / epoch_steps
        logging.info(f"Epoch {epoch}/{num_epochs} 完成 | 平均损失: {epoch_loss:.6f}")
        
        # 保存每个epoch结束后的模型
        checkpoint_path = os.path.join(save_dir, f"checkpoint-epoch-{epoch}")
        os.makedirs(checkpoint_path, exist_ok=True)
        torch.save({
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict() if scaler else None,
            'loss': epoch_loss,
            'vocab_size': vocab_size,
            'd_model': config.get('d_model', 512),
            'n_layers': config.get('n_layers', 6),
            'n_heads': config.get('n_heads', 8),
        }, os.path.join(checkpoint_path, "pytorch_model.bin"))
        logging.info(f"Epoch {epoch} 模型已保存至 {checkpoint_path}")
    
    # 保存最终模型
    final_model_path = os.path.join(save_dir, "final_model")
    os.makedirs(final_model_path, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': vocab_size,
        'd_model': config.get('d_model', 512),
        'n_layers': config.get('n_layers', 6),
        'n_heads': config.get('n_heads', 8),
    }, os.path.join(final_model_path, "pytorch_model.bin"))
    logging.info(f"最终微调模型已保存至 {final_model_path}")

def evaluate(model, val_loader, loss_fn, device='cuda', use_amp=True):
    """评估模型"""
    model.eval()
    total_loss = 0
    total_steps = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            # 安全处理批次数据
            vocab_size = model.embedding.token_embedding.embedding.num_embeddings
            input_ids, attention_mask, labels = safe_process_batch(batch, vocab_size, device)
            
            # 混合精度前向传播
            if use_amp and device != 'cpu':
                device_type = 'cuda' if device.startswith('cuda') else device
                with autocast(device_type=device_type):
                    # 前向传播
                    outputs = model(input_ids, attention_mask)
                    
                    # 计算损失
                    logits = outputs.view(-1, outputs.size(-1))
                    labels = labels.view(-1)
                    loss = loss_fn(logits, labels)
            else:
                # 标准前向传播
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
    parser = argparse.ArgumentParser(description="微调预训练的Transformer模型")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--model_path", type=str, required=True, help="预训练模型路径")
    parser.add_argument("--train_file", type=str, required=True, help="训练数据文件路径(JSONL)")
    parser.add_argument("--val_file", type=str, help="验证数据文件路径(JSONL)")
    parser.add_argument("--batch_size", type=int, default=8, help="批次大小")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮次")
    parser.add_argument("--lr", type=float, default=1e-5, help="学习率")
    parser.add_argument("--max_samples", type=int, help="最大样本数量")
    parser.add_argument("--cuda", action="store_true", help="使用CUDA")
    parser.add_argument("--mps", action="store_true", help="使用MPS (Apple Silicon)")
    parser.add_argument("--amp", action="store_true", help="使用混合精度训练")
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 更新配置
    config['batch_size'] = args.batch_size
    config['num_epochs'] = args.epochs
    config['learning_rate'] = args.lr
    config['save_dir'] = config.get('save_dir', './saved_models/finetuned')
    
    # 设置设备
    device = setup_device(args.cuda, args.mps)
    use_amp = args.amp or config.get('use_amp', False)
    
    # 加载分词器
    tokenizer = load_tokenizer(config.get('tokenizer_dir', 'data/hf_tokenizer'))
    
    # 加载模型
    model = load_model(args.model_path, config, device)
    
    # 加载数据
    train_data = load_jsonl(args.train_file, args.max_samples)
    
    # 划分训练集和验证集（如果未提供验证集文件）
    val_data = None
    if args.val_file:
        val_data = load_jsonl(args.val_file, args.max_samples // 10 if args.max_samples else None)
    elif len(train_data) > 10:
        # 从训练集中划分10%作为验证集
        split_idx = int(len(train_data) * 0.9)
        val_data = train_data[split_idx:]
        train_data = train_data[:split_idx]
        logging.info(f"从训练集划分了 {len(val_data)} 个样本作为验证集")
    
    # 创建数据集
    train_dataset = QADataset(train_data, tokenizer, max_length=config.get('max_seq_len', 512))
    val_dataset = QADataset(val_data, tokenizer, max_length=config.get('max_seq_len', 512)) if val_data else None
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=config.get('num_workers', 2)
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        num_workers=config.get('num_workers', 2)
    ) if val_dataset else None
    
    # 微调模型
    finetune(config, model, train_loader, val_loader, device=device, use_amp=use_amp)

if __name__ == "__main__":
    main()
