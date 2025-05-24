#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
处理对话式指令数据集
"""

import os
import sys
import logging
import argparse

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.preprocessing import prepare_dialogue_instruction_data
from data.tokenizer import Tokenizer

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="处理对话式指令数据集")
    parser.add_argument("--input", type=str, default="data/raw/dialogue_instructions.txt", help="输入文件路径")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="输出目录")
    parser.add_argument("--split_type", type=str, default="paired", choices=["conversation", "line", "paired"], 
                       help="分割类型: 'conversation'单独处理每个对话, 'line'整行处理, 'paired'成对处理")
    parser.add_argument("--output_format", type=str, default="jsonl", choices=["jsonl", "txt"], 
                       help="输出格式: 'jsonl'或'txt'")
    parser.add_argument("--special_tokens", action="store_true", help="是否保留<s>和</s>特殊标记")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="分词器路径，如果指定则更新分词器")
    parser.add_argument("--train_val_split", action="store_true", help="是否划分训练集和验证集")
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 构建输出文件路径
    output_base = os.path.join(args.output_dir, f"dialogue_{args.split_type}")
    output_file = f"{output_base}.{args.output_format}"
    
    # 处理对话数据
    logging.info(f"开始处理对话指令数据: {args.input}")
    logging.info(f"分割类型: {args.split_type}, 输出格式: {args.output_format}, 保留特殊标记: {args.special_tokens}")
    
    sample_count = prepare_dialogue_instruction_data(
        input_file=args.input,
        output_file=output_file,
        split_type=args.split_type,
        output_format=args.output_format,
        add_special_tokens=args.special_tokens
    )
    
    logging.info(f"处理完成，总样本数: {sample_count}")
    
    # 如果需要，更新分词器以包含特殊标记
    if args.tokenizer_path:
        try:
            # 加载现有分词器
            tokenizer = Tokenizer.from_file(args.tokenizer_path)
            logging.info(f"已加载分词器: {args.tokenizer_path}")
            
            # 添加特殊标记（如果尚未存在）
            special_tokens = ["<s>", "</s>"]
            added = False
            
            for token in special_tokens:
                if token not in tokenizer.token2id:
                    tokenizer.add_special_token(token)
                    added = True
                    logging.info(f"已添加特殊标记: {token}")
            
            if added:
                # 保存更新后的分词器
                tokenizer.save(args.tokenizer_path)
                logging.info(f"分词器已更新并保存到: {args.tokenizer_path}")
            else:
                logging.info("分词器已包含所有必要的特殊标记，无需更新")
        except Exception as e:
            logging.error(f"更新分词器时出错: {str(e)}")
    
    # 如果需要，划分训练集和验证集
    if args.train_val_split:
        from data.preprocessing import create_train_val_split
        
        train_file = f"{output_base}_train.{args.output_format}"
        val_file = f"{output_base}_val.{args.output_format}"
        
        create_train_val_split(
            input_file=output_file,
            train_file=train_file,
            val_file=val_file,
            val_ratio=0.1,
            shuffle=True
        )
        
        logging.info(f"已划分训练集和验证集: {train_file}, {val_file}")

if __name__ == "__main__":
    main()
