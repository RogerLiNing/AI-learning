#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
用于加载和处理Wikipedia JSONL格式数据的模块
支持source/completion格式的JSONL数据
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Union
import torch
from torch.utils.data import Dataset

class WikiDataset(Dataset):
    """
    加载和处理Wikipedia格式JSONL数据的数据集类
    
    支持的数据格式:
    1. 标准JSONL，每行是一个JSON对象: {"source": "...", "completion": "..."}
    2. 可配置字段名称，支持不同数据集的字段结构
    """
    
    def __init__(
        self, 
        file_path: str, 
        tokenizer: Any, 
        max_length: int = 512,
        source_field: str = "source",
        completion_field: str = "completion",
        text_field: Optional[str] = None,  # 单字段模式
        add_special_tokens: bool = True,
        max_samples: Optional[int] = None
    ):
        """
        初始化WikiDataset
        
        Args:
            file_path: JSONL文件路径
            tokenizer: 用于编码文本的分词器
            max_length: 最大序列长度
            source_field: 源文本的字段名
            completion_field: 目标文本的字段名
            text_field: 单字段模式的字段名，如果提供，将忽略source_field和completion_field
            add_special_tokens: 是否添加特殊标记
            max_samples: 最大样本数量，None表示不限制
        """
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.source_field = source_field
        self.completion_field = completion_field
        self.text_field = text_field
        self.add_special_tokens = add_special_tokens
        
        # 加载数据
        self.examples = self._load_data(file_path, max_samples)
        logging.info(f"加载了 {len(self.examples)} 个样本，来自 {file_path}")
    
    def _load_data(self, file_path: str, max_samples: Optional[int] = None) -> List[Dict[str, str]]:
        """
        加载数据文件，支持JSON数组和JSONL格式
        
        Args:
            file_path: 数据文件路径（JSON数组或JSONL）
            max_samples: 最大样本数量
        
        Returns:
            包含文本数据的列表
        """
        examples = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # 检查文件格式（JSON数组或JSONL）
                first_char = f.read(1).strip()
                f.seek(0)  # 重置文件指针
                
                # 如果第一个非空字符是'['，则当作 JSON 数组处理
                if first_char == '[':
                    logging.info("检测到JSON数组格式，正在加载整个数组...")
                    try:
                        # 加载整个JSON数组
                        data = json.load(f)
                        logging.info(f"成功加载JSON数组，包含 {len(data)} 个项目")
                        
                        # 限制样本数量
                        if max_samples is not None:
                            data = data[:max_samples]
                        
                        # 处理每个项目
                        for i, item in enumerate(data):
                            self._process_item(item, i, examples)
                    except json.JSONDecodeError as e:
                        logging.error(f"JSON解析错误: {e}")
                # 否则，尝试作为JSONL逻辑来处理
                else:
                    logging.info("尝试使用JSONL格式解析...")
                    for i, line in enumerate(f):
                        if max_samples is not None and len(examples) >= max_samples:
                            break
                        
                        line = line.strip()
                        if not line:
                            continue
                        
                        try:
                            # 解析JSON
                            item = json.loads(line)
                            self._process_item(item, i, examples)
                        except json.JSONDecodeError:
                            logging.warning(f"无法解析JSON，行 {i}: {line[:100]}...")
        except Exception as e:
            logging.error(f"加载文件 {file_path} 时出错: {e}")
        
        return examples
    
    def _process_item(self, item, index, examples):
        """处理单个数据项目并添加到examples列表中"""
        # 检查所需字段是否存在
        if self.text_field is not None:
            # 单字段模式
            if self.text_field in item:
                examples.append({"text": item[self.text_field]})
            else:
                logging.warning(f"样本 {index} 缺少字段 {self.text_field}")
        else:
            # 双字段模式
            if self.source_field in item and self.completion_field in item:
                examples.append({
                    "source": item[self.source_field],
                    "completion": item[self.completion_field]
                })
            else:
                missing_fields = []
                if self.source_field not in item:
                    missing_fields.append(self.source_field)
                if self.completion_field not in item:
                    missing_fields.append(self.completion_field)
                logging.warning(f"样本 {index} 缺少字段: {', '.join(missing_fields)}")
    
    def __len__(self) -> int:
        """返回数据集长度"""
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取指定索引的样本"""
        example = self.examples[idx]
        
        if "text" in example:
            # 单字段模式
            return self._process_single_text(example["text"])
        else:
            # 双字段模式
            return self._process_source_completion(example["source"], example["completion"])
    
    def _process_single_text(self, text: str) -> Dict[str, torch.Tensor]:
        """处理单字段文本"""
        # 使用HFTokenizer的encode方法对文本进行编码
        token_ids = self.tokenizer.encode(text, add_special_tokens=self.add_special_tokens)
        
        # 处理序列长度
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]  # 截断
        
        # 创建输入ID和注意力掩码
        input_ids = torch.zeros(self.max_length, dtype=torch.long)
        attention_mask = torch.zeros(self.max_length, dtype=torch.long)
        
        # 填充数据
        seq_length = len(token_ids)
        input_ids[:seq_length] = torch.tensor(token_ids, dtype=torch.long)
        attention_mask[:seq_length] = 1
        
        # 标签: 向右移动一位，预测下一个词
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]
        labels[-1] = 0  # 最后一个位置无法预测下一个词
        
        # 过滤掉padding位置的损失
        labels = labels * attention_mask
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    
    def _process_source_completion(self, source: str, completion: str) -> Dict[str, torch.Tensor]:
        """处理源文本和目标文本"""
        # 组合文本，加入分隔符
        full_text = source + " " + completion
        
        # 使用encode方法进行编码
        token_ids = self.tokenizer.encode(full_text, add_special_tokens=self.add_special_tokens)
        
        # 处理序列长度
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]  # 截断
        
        # 创建输入ID和注意力掩码
        input_ids = torch.zeros(self.max_length, dtype=torch.long)
        attention_mask = torch.zeros(self.max_length, dtype=torch.long)
        
        # 填充数据
        seq_length = len(token_ids)
        input_ids[:seq_length] = torch.tensor(token_ids, dtype=torch.long)
        attention_mask[:seq_length] = 1
        
        # 获取源文本的token数量
        source_ids = self.tokenizer.encode(source, add_special_tokens=False)
        source_len = len(source_ids)
        
        # 创建标签: 源文本部分设为-100(忽略)，目标文本部分设为对应token ID
        labels = input_ids.clone()
        
        # 如果添加了特殊标记，需要调整source_len
        offset = 1 if self.add_special_tokens else 0
        
        # 源文本部分设为-100(忽略)
        if source_len + offset < self.max_length:
            labels[:source_len + offset] = -100
        
        # 处理padding位置
        padding_positions = attention_mask == 0
        labels[padding_positions] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def load_wiki_dataset(
    file_path: str,
    tokenizer: Any,
    max_length: int = 512,
    source_field: str = "source",
    completion_field: str = "completion",
    text_field: Optional[str] = None,
    add_special_tokens: bool = True,
    max_samples: Optional[int] = None
) -> WikiDataset:
    """
    加载Wikipedia格式的数据集
    
    Args:
        file_path: JSONL文件路径
        tokenizer: 分词器
        max_length: 最大序列长度
        source_field: 源文本字段名
        completion_field: 目标文本字段名
        text_field: 单字段模式的字段名
        add_special_tokens: 是否添加特殊标记
        max_samples: 最大样本数量
    
    Returns:
        WikiDataset实例
    """
    return WikiDataset(
        file_path=file_path,
        tokenizer=tokenizer,
        max_length=max_length,
        source_field=source_field,
        completion_field=completion_field,
        text_field=text_field,
        add_special_tokens=add_special_tokens,
        max_samples=max_samples
    )

# 使用示例
if __name__ == "__main__":
    # 此处仅用于测试
    from data.hf_tokenizer import HFTokenizer
    
    # 加载分词器
    tokenizer = HFTokenizer(pretrained_model_name="bert-base-chinese")
    
    # 加载数据集
    dataset = load_wiki_dataset(
        file_path="data/wiki.jsonl",
        tokenizer=tokenizer,
        max_length=512,
        source_field="source",
        completion_field="completion"
    )
    
    # 打印数据集大小
    print(f"数据集大小: {len(dataset)}")
    
    # 获取第一个样本
    sample = dataset[0]
    print(f"样本输入ID: {sample['input_ids'].shape}")
    print(f"样本注意力掩码: {sample['attention_mask'].shape}")
    print(f"样本标签: {sample['labels'].shape}")
