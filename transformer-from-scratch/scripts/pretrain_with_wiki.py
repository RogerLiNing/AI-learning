#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用Wikipedia数据预训练Transformer模型
支持灵活的JSONL格式，可配置字段名称
"""

import os
import sys
import json
import torch
import logging
import argparse
from tqdm import tqdm
from typing import Dict, Any, Optional
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

# 混合精度训练
import contextlib
from torch.amp import autocast, GradScaler

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 安全处理函数，确保所有ID都在有效范围内
def safe_process_batch(batch, vocab_size, device):
    """安全处理批次，确保input_ids不超出词表范围"""
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    
    # 检查输入ID是否超出词表范围 - 使用更安全的方法检测最大值
    # 避免直接使用.max().item()，这在某些CUDA设备上可能有兼容性问题
    try:
        # 先移到CPU，再获取最大值
        max_input_id = input_ids.detach().cpu().max().item()
        if max_input_id >= vocab_size:
            logging.warning(f"发现输入ID超出词表范围: {max_input_id} >= {vocab_size}")
            # 将超出范围的ID替换为UNK token (通常是1)
            input_ids = torch.clamp(input_ids, max=vocab_size-1)
    except Exception as e:
        logging.warning(f"检查词表范围时出错: {e}")
        # 保险起见，应用截断
        input_ids = torch.clamp(input_ids, max=vocab_size-1)
    
    # 检查并限制标签ID范围，同样使用安全方法
    try:
        # 跳过忽略的标签(-100)
        valid_labels = labels[labels != -100]
        if len(valid_labels) > 0:
            max_label_id = valid_labels.detach().cpu().max().item()
            if max_label_id >= vocab_size:
                logging.warning(f"标签中发现超出范围的token ID: {max_label_id} >= {vocab_size}")
                logging.warning(f"自动将超出范围的标签限制在词表大小范围内")
                # 只对非-100的值进行约束
                labels = torch.where(
                    labels != -100,
                    torch.clamp(labels, 0, vocab_size-1),
                    labels
                )
    except Exception as e:
        logging.warning(f"检查标签范围时出错: {e}")
        # 安全起见，对非-100的标签应用截断
        labels = torch.where(
            labels != -100,
            torch.clamp(labels, 0, vocab_size-1),
            labels
        )
    
    return input_ids, attention_mask, labels

from model.transformer import TransformerLM
from data.hf_tokenizer import HFTokenizer
from data.wiki_dataset import load_wiki_dataset

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pretrain_wiki.log", mode='w')
    ]
)

def setup_device(use_cuda, use_mps):
    """设置设备"""
    if use_cuda and torch.cuda.is_available():
        device = "cuda"
        # 启用cuDNN benchmark模式提高性能
        torch.backends.cudnn.benchmark = True
        logging.info(f"使用设备: {device} - {torch.cuda.get_device_name()}")
        logging.info(f"CUDA版本: {torch.version.cuda}")
        logging.info(f"cuDNN版本: {torch.backends.cudnn.version()}")
        # 显示GPU显存信息
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logging.info(f"GPU显存: {total_mem:.2f} GB")
        logging.info(f"cuDNN benchmark 模式已启用")
    elif use_mps and torch.backends.mps.is_available():
        device = "mps"
        logging.info(f"使用设备: {device} - Apple Silicon")
    else:
        device = "cpu"
        logging.info(f"使用设备: {device}")
    
    return device

def load_tokenizer(config):
    """加载分词器"""
    pretrained_model_name = config.get('pretrained_model_name', 'bert-base-chinese')
    tokenizer_dir = config.get('tokenizer_dir', None)
    
    # 如果指定了本地目录且目录存在，从本地加载
    if tokenizer_dir and os.path.isdir(tokenizer_dir):
        tokenizer = HFTokenizer(pretrained_model_name=tokenizer_dir)
        logging.info(f"从本地加载分词器: {tokenizer_dir}, 词表大小: {tokenizer.vocab_size}")
    else:
        # 否则从预训练模型初始化
        tokenizer = HFTokenizer(pretrained_model_name=pretrained_model_name)
        logging.info(f"初始化HuggingFace分词器: {pretrained_model_name}, 词表大小: {tokenizer.vocab_size}")
        
        # 添加特殊标记
        special_tokens = {
            'bos_token': '[BOS]',
            'eos_token': '[EOS]',
        }
        logging.info(f"添加特殊标记: {special_tokens}")
        tokenizer.add_special_tokens(special_tokens)
        
        # 保存分词器
        if tokenizer_dir:
            os.makedirs(tokenizer_dir, exist_ok=True)
            tokenizer.save(tokenizer_dir)
            logging.info(f"分词器已保存至: {tokenizer_dir}")
    
    return tokenizer

def create_wiki_datasets(config, tokenizer):
    """创建Wikipedia格式的数据集"""
    train_data_path = config.get('train_data_path')
    val_data_path = config.get('val_data_path')
    max_seq_len = config.get('max_seq_len', 512)
    max_train_samples = config.get('max_train_samples', None)
    max_val_samples = config.get('max_val_samples', None)
    
    # 数据集字段配置
    source_field = config.get('source_field', 'source')
    completion_field = config.get('completion_field', 'completion')
    text_field = config.get('text_field', None)  # 单字段模式，如果提供则使用
    
    logging.info(f"加载Wikipedia格式数据，源字段: {source_field}, 目标字段: {completion_field}")
    if text_field:
        logging.info(f"使用单字段模式，字段名: {text_field}")
    
    # 创建训练集
    if not os.path.exists(train_data_path):
        logging.error(f"训练数据文件不存在: {train_data_path}")
        raise FileNotFoundError(f"训练数据文件不存在: {train_data_path}")
    
    train_dataset = load_wiki_dataset(
        file_path=train_data_path,
        tokenizer=tokenizer,
        max_length=max_seq_len,
        source_field=source_field,
        completion_field=completion_field,
        text_field=text_field,
        add_special_tokens=True,
        max_samples=max_train_samples
    )
    logging.info(f"创建训练集，样本数: {len(train_dataset)}")
    
    # 创建验证集
    val_dataset = None
    if val_data_path and os.path.exists(val_data_path):
        val_dataset = load_wiki_dataset(
            file_path=val_data_path,
            tokenizer=tokenizer,
            max_length=max_seq_len,
            source_field=source_field,
            completion_field=completion_field,
            text_field=text_field,
            add_special_tokens=True,
            max_samples=max_val_samples
        )
        logging.info(f"创建验证集，样本数: {len(val_dataset)}")
    
    return train_dataset, val_dataset

def create_data_loader(dataset, batch_size, num_workers):
    """创建数据加载器"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

def create_model(config, vocab_size):
    """创建模型"""
    d_model = config.get('d_model', 512)
    n_layers = config.get('n_layers', 6)
    n_heads = config.get('n_heads', 8)
    d_ff = config.get('d_ff', 2048)
    dropout = config.get('dropout', 0.1)
    max_seq_len = config.get('max_seq_len', 512)
    
    logging.info(f"创建模型: d_model={d_model}, n_layers={n_layers}, n_heads={n_heads}")
    
    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        dropout=dropout,
        max_seq_len=max_seq_len
    )
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logging.info(f"模型参数数量: {total_params:,}")
    logging.info(f"模型总参数: {total_params:,} (可训练: {trainable_params:,})")
    
    return model

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """创建带有预热的线性学习率调度器"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)

def train(config, model, train_loader, val_loader=None, device='cuda', use_amp=True):
    """训练模型"""
    # 获取词表大小
    embedding_layer = model.embedding.token_embedding.embedding
    vocab_size = embedding_layer.num_embeddings
    logging.info(f"模型词表大小: {vocab_size}")
    
    # 损失函数 - 忽略-100标签（用于处理源文本部分和padding）
    num_epochs = config.get('num_epochs', 10)
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
    logging_steps = config.get('logging_steps', 100)
    save_steps = config.get('save_steps', 1000)
    eval_steps = config.get('eval_steps', 500)
    
    # 保存目录
    save_dir = config.get('save_dir', './saved_models/wiki_pretrain')
    os.makedirs(save_dir, exist_ok=True)
    
    # 训练状态
    global_step = 0
    train_losses = []
    
    # 优化器
    optimizer = AdamW(
        model.parameters(),
        lr=config.get('learning_rate', 5e-5),
        weight_decay=config.get('weight_decay', 0.01)
    )
    
    # 学习率调度器
    total_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
    warmup_steps = int(total_steps * config.get('warmup_ratio', 0.1))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )
    
    logging.info(f"开始训练... 设备: {device}, 混合精度: {use_amp if device != 'cpu' else False}")
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0
        epoch_steps = 0
        
        # 创建进度条
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
        
        for step, batch in enumerate(progress_bar):
            # 安全处理批次数据，确保所有ID都在有效范围内
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
                    
                    logging.info(f"模型已保存至 {checkpoint_path}")
                
                # 评估模型
                if val_loader and global_step % eval_steps == 0:
                    val_loss = evaluate(model, val_loader, device, use_amp)
                    logging.info(f"Step {global_step} | 验证损失: {val_loss:.4f}")
                    model.train()  # 切回训练模式
                
                # 释放GPU缓存
                if device == 'cuda':
                    torch.cuda.empty_cache()
        
        # 每个epoch结束后的平均损失
        epoch_loss = epoch_loss / epoch_steps
        train_losses.append(epoch_loss)
        logging.info(f"Epoch {epoch}/{num_epochs} 完成 | 平均损失: {epoch_loss:.4f}")
        
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
    logging.info(f"最终模型已保存至 {final_model_path}")
    
    return train_losses

def evaluate(model, val_loader, device='cuda', use_amp=True):
    """评估模型"""
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    total_loss = 0
    total_steps = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            # 安全处理批次数据，确保所有ID都在有效范围内
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
    parser = argparse.ArgumentParser(description="使用Wikipedia数据预训练Transformer模型")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--cuda", action="store_true", help="使用CUDA")
    parser.add_argument("--mps", action="store_true", help="使用MPS (Apple Silicon)")
    parser.add_argument("--amp", action="store_true", help="使用混合精度训练")
    parser.add_argument("--source_field", type=str, help="源文本字段名称，覆盖配置文件中的设置")
    parser.add_argument("--completion_field", type=str, help="目标文本字段名称，覆盖配置文件中的设置")
    parser.add_argument("--text_field", type=str, help="单字段模式的字段名称，覆盖配置文件中的设置")
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 命令行参数覆盖配置文件
    if args.source_field:
        config['source_field'] = args.source_field
    if args.completion_field:
        config['completion_field'] = args.completion_field
    if args.text_field:
        config['text_field'] = args.text_field
    
    # 设置设备
    device = setup_device(args.cuda, args.mps)
    use_amp = args.amp or config.get('use_amp', False)
    if use_amp and device != 'cpu':
        logging.info(f"启用混合精度训练 (AMP)")
    
    # 加载分词器
    tokenizer = load_tokenizer(config)
    
    # 创建数据集
    train_dataset, val_dataset = create_wiki_datasets(config, tokenizer)
    
    # 创建数据加载器
    batch_size = config.get('batch_size', 32)
    num_workers = config.get('num_workers', 4)
    train_loader = create_data_loader(train_dataset, batch_size, num_workers)
    val_loader = None
    if val_dataset:
        val_loader = create_data_loader(val_dataset, batch_size, num_workers)
    
    # 创建模型
    model = create_model(config, tokenizer.vocab_size)
    model = model.to(device)
    
    # 训练模型
    train(config, model, train_loader, val_loader, device=device, use_amp=use_amp)

if __name__ == "__main__":
    main()
