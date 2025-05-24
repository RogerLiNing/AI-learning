import os
import json
import logging
from typing import List, Dict, Optional, Union, Any

from transformers import AutoTokenizer, PreTrainedTokenizerFast

class HFTokenizer:
    """
    HuggingFace分词器的包装类，使其接口与项目自定义分词器兼容
    """
    def __init__(self, 
                 pretrained_model_name: str = "bert-base-chinese",
                 cache_dir: Optional[str] = None,
                 add_special_tokens: bool = True):
        """
        初始化HuggingFace分词器包装类
        
        参数:
            pretrained_model_name: 预训练模型名称或路径
            cache_dir: 缓存目录
            add_special_tokens: 是否在编码时添加特殊标记
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, 
            cache_dir=cache_dir
        )
        
        # 映射自定义属性
        self.vocab_size = len(self.tokenizer)
        self.pad_id = self.tokenizer.pad_token_id
        self.unk_id = self.tokenizer.unk_token_id
        self.add_special_tokens = add_special_tokens
        
        # 确保分词器具有必要的特殊标记
        self._ensure_special_tokens()
        
        logging.info(f"初始化HuggingFace分词器: {pretrained_model_name}, 词表大小: {self.vocab_size}")
    
    def _ensure_special_tokens(self):
        """确保分词器具有必要的特殊标记"""
        special_tokens = {}
        
        # 检查是否存在必要的特殊标记
        if not self.tokenizer.pad_token:
            special_tokens['pad_token'] = '[PAD]'
        
        if not self.tokenizer.unk_token:
            special_tokens['unk_token'] = '[UNK]'
        
        if not self.tokenizer.bos_token:
            special_tokens['bos_token'] = '[BOS]'
        
        if not self.tokenizer.eos_token:
            special_tokens['eos_token'] = '[EOS]'
        
        # 添加特殊标记
        if special_tokens:
            self.tokenizer.add_special_tokens(special_tokens)
            logging.info(f"添加特殊标记: {special_tokens}")
        
        # 添加对话特殊标记
        dialog_tokens = {'s_start': '<s>', 's_end': '</s>'}
        if '<s>' not in self.tokenizer.get_vocab() or '</s>' not in self.tokenizer.get_vocab():
            dialog_special_tokens = {'additional_special_tokens': ['<s>', '</s>']}
            self.tokenizer.add_special_tokens(dialog_special_tokens)
            logging.info("添加对话特殊标记: <s>, </s>")
    
    def tokenize(self, text: str) -> List[str]:
        """
        将文本分词为token列表
        
        参数:
            text: 输入文本
            
        返回:
            token列表
        """
        return self.tokenizer.tokenize(text)
    
    def encode(self, text: str, add_special_tokens: Optional[bool] = None, max_length: Optional[int] = None) -> List[int]:
        """
        将文本编码为token ID列表
        
        参数:
            text: 输入文本
            add_special_tokens: 是否添加特殊标记，如果为None则使用初始化时的设置
            max_length: 最大长度，如果为None则不限制长度
            
        返回:
            token ID列表
        """
        if add_special_tokens is None:
            add_special_tokens = self.add_special_tokens
        
        # 如果设置了max_length，使用截断和填充
        if max_length is not None:
            encoding = self.tokenizer.encode(
                text,
                add_special_tokens=add_special_tokens,
                max_length=max_length,
                truncation=True,
                padding='max_length'
            )
            
            # 创建注意力掩码
            attention_mask = [1] * len(encoding)
            
            return encoding
        else:
            # 否则，只进行编码
            return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        将token ID列表解码为文本
        
        参数:
            token_ids: token ID列表
            skip_special_tokens: 是否跳过特殊标记
            
        返回:
            解码后的文本
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def save(self, save_path: str):
        """
        保存分词器
        
        参数:
            save_path: 保存路径
        """
        # 创建目录
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 保存分词器
        self.tokenizer.save_pretrained(save_path)
        
        # 保存自定义信息
        info_path = os.path.join(os.path.dirname(save_path), "tokenizer_info.json")
        info = {
            "type": "hf_tokenizer",
            "pretrained_model_name": self.tokenizer.name_or_path,
            "vocab_size": self.vocab_size,
            "add_special_tokens": self.add_special_tokens
        }
        
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        
        logging.info(f"分词器已保存至: {save_path}")
    
    @classmethod
    def from_file(cls, path: str):
        """
        从文件加载分词器
        
        参数:
            path: 分词器路径
            
        返回:
            HFTokenizer实例
        """
        # 从HuggingFace格式加载
        tokenizer = cls()
        tokenizer.tokenizer = AutoTokenizer.from_pretrained(path)
        
        # 更新属性
        tokenizer.vocab_size = len(tokenizer.tokenizer)
        tokenizer.pad_id = tokenizer.tokenizer.pad_token_id
        tokenizer.unk_id = tokenizer.tokenizer.unk_token_id
        
        # 尝试加载自定义信息
        info_path = os.path.join(os.path.dirname(path), "tokenizer_info.json")
        if os.path.exists(info_path):
            with open(info_path, 'r', encoding='utf-8') as f:
                info = json.load(f)
                tokenizer.add_special_tokens = info.get('add_special_tokens', True)
        
        logging.info(f"从 {path} 加载HuggingFace分词器，词表大小: {tokenizer.vocab_size}")
        
        return tokenizer
    
    @property
    def token2id(self) -> Dict[str, int]:
        """获取token到ID的映射"""
        return self.tokenizer.get_vocab()
    
    @property
    def id2token(self) -> Dict[int, str]:
        """获取ID到token的映射"""
        vocab = self.tokenizer.get_vocab()
        return {v: k for k, v in vocab.items()}
