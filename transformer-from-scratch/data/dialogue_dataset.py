import os
import json
import torch
import logging
from typing import List, Dict, Optional, Tuple, Union
from torch.utils.data import Dataset

from .dataset import TextDataset
from .tokenizer import Tokenizer

class DialogueDataset(TextDataset):
    """
    对话式指令数据集，用于处理成对的问答数据
    """
    def __init__(self, 
                 texts: List[Dict],
                 tokenizer: Tokenizer,
                 max_length: int = 512,
                 add_special_tokens: bool = True):
        """
        初始化对话数据集
        
        参数:
            texts: 对话字典列表，每个字典包含'query'和'response'字段
            tokenizer: 分词器实例
            max_length: 最大序列长度
            add_special_tokens: 是否添加特殊标记
        """
        # 我们不直接调用父类的__init__，因为我们需要不同的处理方式
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_special_tokens = add_special_tokens
        
        logging.info(f"创建对话数据集，样本数: {len(texts)}")
    
    def __getitem__(self, idx):
        """获取数据集中的一个样本"""
        item = self.texts[idx]
        
        # 提取问题和回答
        query = item['query']
        response = item['response']
        
        # 标记化
        s_start = '<s>'  # 对话开始标记
        s_end = '</s>'   # 对话结束标记
        
        # 构造输入序列: <s>问题</s> <s>回答</s>
        if self.add_special_tokens:
            full_text = f"{s_start}{query}{s_end} {s_start}{response}{s_end}"
        else:
            full_text = f"{query} {response}"
        
        # 编码全文
        input_ids = self.tokenizer.encode(
            full_text, 
            add_special_tokens=False,  # 我们已经手动添加了特殊标记
            max_length=self.max_length
        )
        
        # 对于自回归模型，标签是输入序列向右移动一位
        labels = input_ids[1:] + [self.tokenizer.pad_id]
        
        # 创建注意力掩码
        attention_mask = [1] * len(input_ids)
        
        # 确保长度等于max_length
        if len(input_ids) < self.max_length:
            # 填充
            padding_length = self.max_length - len(input_ids)
            input_ids = input_ids + [self.tokenizer.pad_id] * padding_length
            labels = labels + [self.tokenizer.pad_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length
        else:
            # 截断
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
            
        return {
            'input_ids': torch.tensor(input_ids),
            'labels': torch.tensor(labels),
            'attention_mask': torch.tensor(attention_mask)
        }
    
    @classmethod
    def from_file(cls, file_path: str, tokenizer: Tokenizer, **kwargs):
        """
        从JSONL文件加载对话数据集
        
        参数:
            file_path: JSONL文件路径，每行是一个包含'query'和'response'字段的JSON对象
            tokenizer: 分词器实例
            **kwargs: 其他参数传递给构造函数
            
        返回:
            DialogueDataset实例
        """
        items = []
        
        # 加载JSONL文件
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        if 'query' in data and 'response' in data:
                            items.append(data)
                    except json.JSONDecodeError:
                        logging.warning(f"无法解析行: {line}")
                        continue
        
        logging.info(f"从文件 {file_path} 加载了 {len(items)} 个对话样本")
        
        return cls(items, tokenizer, **kwargs)
