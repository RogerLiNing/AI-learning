import os
import json
import torch
import logging
from typing import List, Dict, Optional, Tuple, Union, Any
from torch.utils.data import Dataset

from .dataset import TextDataset
from .tokenizer import Tokenizer

# 支持HFTokenizer类型
TokenizerType = Union[Tokenizer, Any]

class ConversationDataset(TextDataset):
    """
    对话数据集，用于处理单行完整对话数据
    """
    def __init__(self, 
                 texts: List[str],
                 tokenizer: TokenizerType,
                 max_length: int = 512,
                 add_special_tokens: bool = False):  # 默认为False，因为数据中已包含特殊标记
        """
        初始化对话数据集
        
        参数:
            texts: 对话文本列表，每个文本是一个完整对话
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
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        """获取数据集中的一个样本"""
        text = self.texts[idx]
        
        # 编码文本
        input_ids = self.tokenizer.encode(
            text, 
            add_special_tokens=self.add_special_tokens,
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
    def from_file(cls, file_path: str, tokenizer: TokenizerType, **kwargs):
        """
        从文本文件加载对话数据集
        
        参数:
            file_path: 文本文件路径，每行是一个完整对话
            tokenizer: 分词器实例，可以是Tokenizer或HFTokenizer
            **kwargs: 其他参数传递给构造函数
            
        返回:
            ConversationDataset实例
        """
        texts = []
        
        # 加载文本文件
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    texts.append(line.strip())
        
        logging.info(f"从文件 {file_path} 加载了 {len(texts)} 个对话样本")
        
        return cls(texts, tokenizer, **kwargs)
