#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
处理JSONL格式的对话式指令数据集
"""

import os
import sys
import json
import logging
import argparse
import re
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.tokenizer import Tokenizer

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def process_jsonl_dialogue(input_file, output_file, split_type='conversation', keep_special_tokens=True):
    """
    处理JSONL格式的对话数据
    
    参数:
        input_file: 输入JSONL文件路径
        output_file: 输出文件路径
        split_type: 分割类型，'conversation'表示每个<s>...</s>作为一个样本
        keep_special_tokens: 是否保留<s>和</s>特殊标记
    
    返回:
        处理的样本数
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    processed_samples = []
    total_conversations = 0
    
    # 读取JSONL文件
    with open(input_file, 'r', encoding='utf-8') as f_in:
        for line in tqdm(f_in, desc=f"处理 {os.path.basename(input_file)}"):
            if not line.strip():
                continue
                
            try:
                # 解析JSON对象
                data = json.loads(line)
                if 'text' in data:
                    text = data['text']
                    
                    # 提取所有<s>...</s>对话段落
                    conversations = re.findall(r'<s>(.*?)</s>', text)
                    total_conversations += len(conversations)
                    
                    # 根据分割类型处理
                    if split_type == 'conversation':
                        # 每个<s>...</s>作为一个单独样本
                        for conv in conversations:
                            if keep_special_tokens:
                                processed_samples.append(f"<s>{conv}</s>")
                            else:
                                processed_samples.append(conv)
                    else:  # 'document'
                        # 整个文档作为一个样本
                        processed_samples.append(text)
            except json.JSONDecodeError:
                logging.warning(f"无法解析行: {line[:50]}...")
                continue
    
    # 写入处理后的样本
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for sample in processed_samples:
            f_out.write(sample + '\n')
    
    logging.info(f"处理完成: {input_file} -> {output_file}")
    logging.info(f"总JSON对象数: {len(processed_samples) if split_type == 'document' else 'N/A'}")
    logging.info(f"总对话数: {total_conversations}")
    logging.info(f"输出样本数: {len(processed_samples)}")
    
    return len(processed_samples)

def create_train_val_split(input_file, train_file, val_file, val_ratio=0.1, shuffle=True):
    """
    将数据集分割为训练集和验证集
    """
    # 读取所有样本
    with open(input_file, 'r', encoding='utf-8') as f:
        samples = [line.strip() for line in f if line.strip()]
    
    # 打乱数据（可选）
    if shuffle:
        import random
        random.shuffle(samples)
    
    # 计算分割点
    split_idx = int(len(samples) * (1 - val_ratio))
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]
    
    # 写入训练集
    with open(train_file, 'w', encoding='utf-8') as f:
        for sample in train_samples:
            f.write(sample + '\n')
    
    # 写入验证集
    with open(val_file, 'w', encoding='utf-8') as f:
        for sample in val_samples:
            f.write(sample + '\n')
    
    logging.info(f"数据集分割完成: {input_file} -> {train_file} ({len(train_samples)}行), {val_file} ({len(val_samples)}行)")
    
    return train_file, val_file

def update_tokenizer_with_special_tokens(tokenizer_path):
    """
    更新分词器以包含对话所需的特殊标记
    """
    try:
        # 加载现有分词器
        tokenizer = Tokenizer.from_file(tokenizer_path)
        logging.info(f"已加载分词器: {tokenizer_path}")
        
        # 检查特殊标记是否已存在
        special_tokens = ["<s>", "</s>"]
        all_exist = True
        
        for token in special_tokens:
            if token not in tokenizer.token2id:
                all_exist = False
                break
        
        if all_exist:
            logging.info("分词器已包含所有必要的特殊标记，无需更新")
            return tokenizer
        
        # 创建新的分词器，包含所有必要的特殊标记
        updated_special_tokens = {
            'pad': '[PAD]',
            'unk': '[UNK]',
            'bos': '[BOS]',
            'eos': '[EOS]',
            'mask': '[MASK]',
            's_start': '<s>',  # 对话开始标记
            's_end': '</s>'    # 对话结束标记
        }
        
        # 创建新的分词器
        new_tokenizer = Tokenizer(
            vocab_size=tokenizer.vocab_size,
            special_tokens=updated_special_tokens
        )
        
        # 将原分词器的词表复制到新分词器（排除特殊标记）
        for token, token_id in tokenizer.token2id.items():
            if token not in new_tokenizer.token2id:
                new_tokenizer._add_token(token)
        
        # 保存更新后的分词器
        new_tokenizer.save(tokenizer_path)
        logging.info(f"分词器已更新并保存到: {tokenizer_path}")
        
        return new_tokenizer
    except Exception as e:
        logging.error(f"更新分词器时出错: {str(e)}")
        return None

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="处理JSONL格式的对话式指令数据集")
    parser.add_argument("--input", type=str, default="data/processed/pretrain.jsonl", 
                        help="输入JSONL文件路径")
    parser.add_argument("--output_dir", type=str, default="data/processed", 
                        help="输出目录")
    parser.add_argument("--split_type", type=str, default="conversation", 
                        choices=["conversation", "document"], 
                        help="分割类型: 'conversation'单独处理每个对话, 'document'整行处理")
    parser.add_argument("--special_tokens", action="store_true", default=True,
                        help="是否保留<s>和</s>特殊标记")
    parser.add_argument("--tokenizer_path", type=str, 
                        default="data/tokenizer/tokenizer_dialogue.json", 
                        help="分词器路径，如果指定则更新分词器")
    parser.add_argument("--train_val_split", action="store_true", default=True,
                        help="是否划分训练集和验证集")
    parser.add_argument("--val_ratio", type=float, default=0.1, 
                        help="验证集比例")
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 构建输出文件路径
    output_base = os.path.join(args.output_dir, f"dialogue_{args.split_type}")
    output_file = f"{output_base}.txt"
    
    # 处理对话数据
    logging.info(f"开始处理对话指令数据: {args.input}")
    logging.info(f"分割类型: {args.split_type}, 保留特殊标记: {args.special_tokens}")
    
    sample_count = process_jsonl_dialogue(
        input_file=args.input,
        output_file=output_file,
        split_type=args.split_type,
        keep_special_tokens=args.special_tokens
    )
    
    logging.info(f"处理完成，总样本数: {sample_count}")
    
    # 如果需要，更新分词器以包含特殊标记
    if args.tokenizer_path:
        update_tokenizer_with_special_tokens(args.tokenizer_path)
    
    # 如果需要，划分训练集和验证集
    if args.train_val_split:
        train_file = f"{output_base}_train.txt"
        val_file = f"{output_base}_val.txt"
        
        create_train_val_split(
            input_file=output_file,
            train_file=train_file,
            val_file=val_file,
            val_ratio=args.val_ratio,
            shuffle=True
        )
        
        logging.info(f"已划分训练集和验证集: {train_file}, {val_file}")
        
        # 返回文件路径，以便后续使用
        return train_file, val_file
    
    return output_file, None

if __name__ == "__main__":
    main()
