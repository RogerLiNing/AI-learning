#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用预训练的Transformer模型生成文本
"""

import os
import sys
import torch
import logging
import argparse
import json
from typing import List, Optional

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.transformer import TransformerLM
from data.hf_tokenizer import HFTokenizer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
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
    model.eval()  # 设置为评估模式
    
    return model

def generate_text(model, tokenizer, prompt, max_length=100, temperature=1.0, top_k=50, top_p=0.9, device='cuda'):
    """生成文本"""
    # 编码提示文本
    prompt_ids = tokenizer.encode(prompt)
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long).to(device)
    
    # 生成文本
    with torch.no_grad():
        output_ids = model.generate(
            prompt_tensor,
            max_len=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
    
    # 解码生成的文本
    generated_text = tokenizer.decode(output_ids[0].tolist())
    
    return generated_text

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="使用预训练的Transformer模型生成文本")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--model_path", type=str, required=True, help="模型检查点路径")
    parser.add_argument("--prompt", type=str, default="你好，", help="生成的提示文本")
    parser.add_argument("--max_length", type=int, default=100, help="生成的最大长度")
    parser.add_argument("--temperature", type=float, default=1.0, help="生成的温度参数")
    parser.add_argument("--top_k", type=int, default=50, help="top-k采样参数")
    parser.add_argument("--top_p", type=float, default=0.9, help="top-p采样参数")
    parser.add_argument("--cuda", action="store_true", help="使用CUDA")
    parser.add_argument("--mps", action="store_true", help="使用MPS (Apple Silicon)")
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 设置设备
    device = setup_device(args.cuda, args.mps)
    
    # 加载分词器
    tokenizer = load_tokenizer(config.get('tokenizer_dir', 'data/hf_tokenizer'))
    
    # 加载模型
    model = load_model(args.model_path, config, device)
    
    # 生成文本
    generated_text = generate_text(
        model, 
        tokenizer, 
        args.prompt, 
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=device
    )
    
    print("\n" + "="*50)
    print(f"提示: {args.prompt}")
    print("="*50)
    print(f"生成的文本: \n{generated_text}")
    print("="*50)

if __name__ == "__main__":
    main()
