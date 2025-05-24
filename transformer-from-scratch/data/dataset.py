import os
import json
import torch
import random
import logging
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Tuple, Union
from .tokenizer import Tokenizer


class TextDataset(Dataset):
    """
    用于Transformer模型的文本数据集
    """
    def __init__(self, 
                 texts: List[str],
                 tokenizer: Tokenizer,
                 max_length: int = 512,
                 add_special_tokens: bool = True):
        """
        初始化文本数据集
        
        参数:
            texts: 文本列表
            tokenizer: 分词器实例
            max_length: 最大序列长度
            add_special_tokens: 是否添加特殊标记
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_special_tokens = add_special_tokens
        
        logging.info(f"创建数据集，样本数: {len(texts)}")
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # 编码文本
        input_ids = self.tokenizer.encode(
            text, 
            add_special_tokens=self.add_special_tokens, 
            max_length=self.max_length
        )
        
        # 创建注意力掩码（1表示非填充位置，0表示填充位置）
        attention_mask = [1] * len(input_ids)
        
        # 确保长度等于max_length
        if len(input_ids) < self.max_length:
            # 填充
            padding_length = self.max_length - len(input_ids)
            input_ids = input_ids + [self.tokenizer.pad_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length
        else:
            # 截断
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
            
        return {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask)
        }
    
    @classmethod
    def from_file(cls, file_path: str, tokenizer: Tokenizer, **kwargs):
        """
        从文件加载数据集
        
        参数:
            file_path: 文件路径，可以是txt文件（每行一个样本）或jsonl文件（每行一个JSON对象）
            tokenizer: 分词器实例
            **kwargs: 其他参数传递给构造函数
            
        返回:
            TextDataset实例
        """
        texts = []
        
        # 根据文件扩展名处理不同格式
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
        elif ext in ['.json', '.jsonl']:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        # 假设JSON对象有一个'text'字段
                        if 'text' in data:
                            texts.append(data['text'])
        else:
            raise ValueError(f"不支持的文件格式: {ext}")
            
        logging.info(f"从文件 {file_path} 加载了 {len(texts)} 个样本")
        
        return cls(texts, tokenizer, **kwargs)


class MLMDataset(TextDataset):
    """
    掩码语言模型(Masked Language Modeling)数据集
    
    用于预训练Transformer模型
    """
    def __init__(self, 
                 texts: List[str],
                 tokenizer: Tokenizer,
                 max_length: int = 512,
                 mlm_probability: float = 0.15):
        """
        初始化MLM数据集
        
        参数:
            texts: 文本列表
            tokenizer: 分词器实例
            max_length: 最大序列长度
            mlm_probability: 掩码比例
        """
        super().__init__(texts, tokenizer, max_length, add_special_tokens=True)
        self.mlm_probability = mlm_probability
        
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # 编码文本
        input_ids = self.tokenizer.encode(
            text, 
            add_special_tokens=True, 
            max_length=self.max_length
        )
        
        # 创建注意力掩码
        attention_mask = [1] * len(input_ids)
        
        # 填充
        if len(input_ids) < self.max_length:
            padding_length = self.max_length - len(input_ids)
            input_ids = input_ids + [self.tokenizer.pad_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length
        else:
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
        
        # 创建用于MLM的标签（复制输入ID）
        labels = input_ids.copy()
        
        # 应用掩码
        masked_indices = self._get_mask_indices(input_ids, attention_mask)
        input_ids = self._apply_mask(input_ids, masked_indices)
        
        return {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask),
            'labels': torch.tensor(labels)
        }
    
    def _get_mask_indices(self, input_ids: List[int], attention_mask: List[int]) -> List[int]:
        """
        获取需要掩码的索引
        
        参数:
            input_ids: 输入ID列表
            attention_mask: 注意力掩码列表
            
        返回:
            要掩码的索引列表
        """
        # 特殊标记的索引集合
        special_tokens_mask = [
            1 if id in [self.tokenizer.pad_id, self.tokenizer.bos_id, self.tokenizer.eos_id] else 0
            for id in input_ids
        ]
        
        # 只在非特殊标记和非填充位置应用掩码
        masked_indices = []
        for i in range(len(input_ids)):
            if attention_mask[i] == 1 and special_tokens_mask[i] == 0:
                if random.random() < self.mlm_probability:
                    masked_indices.append(i)
                    
        return masked_indices
    
    def _apply_mask(self, input_ids: List[int], masked_indices: List[int]) -> List[int]:
        """
        应用掩码策略
        
        - 80%的情况下，将token替换为[MASK]
        - 10%的情况下，将token替换为随机token
        - 10%的情况下，保持token不变
        
        参数:
            input_ids: 输入ID列表
            masked_indices: 要掩码的索引列表
            
        返回:
            掩码后的输入ID列表
        """
        # 创建输入ID的副本
        masked_input_ids = input_ids.copy()
        
        # 获取词表大小（排除特殊标记）
        vocab_size = len(self.tokenizer.token2id)
        
        for index in masked_indices:
            # 80%的情况：替换为[MASK]
            if random.random() < 0.8:
                masked_input_ids[index] = self.tokenizer.mask_id
            # 10%的情况：替换为随机token
            elif random.random() < 0.5:  # 0.8 + (1-0.8)*0.5 = 0.9
                masked_input_ids[index] = random.randint(0, vocab_size - 1)
            # 10%的情况：保持不变
                
        return masked_input_ids


class CLMDataset(TextDataset):
    """
    因果语言模型(Causal Language Modeling)数据集
    
    用于自回归语言模型预训练
    """
    def __init__(self, 
                 texts: List[str],
                 tokenizer: Tokenizer,
                 max_length: int = 512):
        """
        初始化CLM数据集
        
        参数:
            texts: 文本列表
            tokenizer: 分词器实例
            max_length: 最大序列长度
        """
        super().__init__(texts, tokenizer, max_length, add_special_tokens=True)
        
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # 编码文本
        input_ids = self.tokenizer.encode(
            text, 
            add_special_tokens=True, 
            max_length=self.max_length
        )
        
        # 创建注意力掩码
        attention_mask = [1] * len(input_ids)
        
        # 填充或截断
        if len(input_ids) < self.max_length:
            padding_length = self.max_length - len(input_ids)
            input_ids = input_ids + [self.tokenizer.pad_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length
        else:
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
        
        # 对于CLM，标签与输入相同，但将填充位置设为-100（忽略这些位置的损失计算）
        labels = [-100 if mask == 0 else input_id for input_id, mask in zip(input_ids, attention_mask)]
        
        return {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask),
            'labels': torch.tensor(labels)
        }


class ClassificationDataset(Dataset):
    """
    文本分类数据集
    
    用于微调Transformer模型进行分类任务
    """
    def __init__(self, 
                 texts: List[str],
                 labels: List[int],
                 tokenizer: Tokenizer,
                 max_length: int = 512,
                 label_map: Optional[Dict[str, int]] = None):
        """
        初始化分类数据集
        
        参数:
            texts: 文本列表
            labels: 标签列表（整数）
            tokenizer: 分词器实例
            max_length: 最大序列长度
            label_map: 标签映射，将文本标签映射到整数
        """
        assert len(texts) == len(labels), "文本和标签数量必须相同"
        
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_map = label_map
        
        logging.info(f"创建分类数据集，样本数: {len(texts)}")
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # 编码文本
        encoding = self.tokenizer.encode(
            text, 
            add_special_tokens=True, 
            max_length=self.max_length
        )
        
        # 创建注意力掩码
        attention_mask = [1] * len(encoding)
        
        # 填充或截断
        if len(encoding) < self.max_length:
            padding_length = self.max_length - len(encoding)
            encoding = encoding + [self.tokenizer.pad_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length
        else:
            encoding = encoding[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
            
        return {
            'input_ids': torch.tensor(encoding),
            'attention_mask': torch.tensor(attention_mask),
            'label': torch.tensor(label)
        }
    
    @classmethod
    def from_file(cls, file_path: str, tokenizer: Tokenizer, text_field: str = 'text', 
                 label_field: str = 'label', **kwargs):
        """
        从文件加载分类数据集
        
        参数:
            file_path: 文件路径，支持jsonl格式
            tokenizer: 分词器实例
            text_field: 文本字段名
            label_field: 标签字段名
            **kwargs: 其他参数传递给构造函数
            
        返回:
            ClassificationDataset实例
        """
        texts = []
        labels = []
        label_set = set()
        
        # 加载数据
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    if text_field in data and label_field in data:
                        texts.append(data[text_field])
                        label = data[label_field]
                        labels.append(label)
                        label_set.add(label)
        
        # 如果标签是文本形式，创建标签映射
        if any(isinstance(label, str) for label in labels):
            label_map = {label: i for i, label in enumerate(sorted(label_set))}
            labels = [label_map[label] for label in labels]
        else:
            label_map = None
            
        logging.info(f"从文件 {file_path} 加载了 {len(texts)} 个样本，类别数: {len(label_set)}")
        
        return cls(texts, labels, tokenizer, label_map=label_map, **kwargs)


def create_dataloader(dataset: Dataset, batch_size: int, shuffle: bool = True, 
                     num_workers: int = 4, pin_memory: bool = True):
    """
    创建数据加载器
    
    参数:
        dataset: 数据集实例
        batch_size: 批次大小
        shuffle: 是否打乱数据
        num_workers: 数据加载线程数
        pin_memory: 是否将数据固定在内存中（加速GPU训练）
        
    返回:
        DataLoader实例
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
