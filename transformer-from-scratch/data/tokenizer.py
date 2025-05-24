import os
import json
import torch
import logging
from collections import Counter
from typing import List, Dict, Optional, Tuple

class Tokenizer:
    """
    简单的分词器实现，支持BPE(Byte-Pair Encoding)算法
    """
    def __init__(self, vocab_size=30000, special_tokens=None):
        """
        初始化分词器
        
        参数:
            vocab_size: 词表大小
            special_tokens: 特殊标记字典，例如 {'pad': '[PAD]', 'unk': '[UNK]', 'bos': '[BOS]', 'eos': '[EOS]', 'mask': '[MASK]'}
        """
        self.vocab_size = vocab_size
        self.token2id = {}
        self.id2token = {}
        
        # 设置默认特殊标记
        self.special_tokens = {
            'pad': '[PAD]',
            'unk': '[UNK]',
            'bos': '[BOS]',
            'eos': '[EOS]',
            'mask': '[MASK]'
        }
        
        # 更新用户提供的特殊标记
        if special_tokens:
            self.special_tokens.update(special_tokens)
            
        # 将特殊标记添加到词表
        for token in self.special_tokens.values():
            self._add_token(token)
            
        # 记录特殊标记ID，便于快速访问
        self.pad_id = self.token2id.get(self.special_tokens['pad'], 0)
        self.unk_id = self.token2id.get(self.special_tokens['unk'], 1)
        self.bos_id = self.token2id.get(self.special_tokens['bos'], 2)
        self.eos_id = self.token2id.get(self.special_tokens['eos'], 3)
        self.mask_id = self.token2id.get(self.special_tokens['mask'], 4)
        
    def _add_token(self, token):
        """将token添加到词表"""
        if token not in self.token2id:
            id = len(self.token2id)
            self.token2id[token] = id
            self.id2token[id] = token
            return id
        return self.token2id[token]
        
    def build_vocab(self, texts: List[str], min_freq: int = 2):
        """
        从文本语料库构建词表
        
        参数:
            texts: 文本列表
            min_freq: 最小出现频率
        """
        # 构建词频统计
        counter = Counter()
        
        # 按字符或单词分割文本并统计频率
        for text in texts:
            # 这里简单按空格分词，实际应用中可能需要更复杂的分词策略
            tokens = text.split()
            counter.update(tokens)
            
        # 按频率排序并保留频率大于min_freq的词
        sorted_tokens = [token for token, count in counter.most_common() 
                        if count >= min_freq]
        
        # 确保不超过词表大小限制（减去特殊标记数量）
        remaining_size = self.vocab_size - len(self.token2id)
        sorted_tokens = sorted_tokens[:remaining_size]
        
        # 将词添加到词表
        for token in sorted_tokens:
            self._add_token(token)
            
        logging.info(f"词表构建完成，大小: {len(self.token2id)}")
        
    def tokenize(self, text: str) -> List[str]:
        """
        将文本分词
        
        参数:
            text: 输入文本
            
        返回:
            token列表
        """
        # 这里使用简单的空格分词，实际应用中可能需要更复杂的分词策略
        return text.split()
    
    def encode(self, text: str, add_special_tokens: bool = True, max_length: Optional[int] = None) -> List[int]:
        """
        将文本编码为ID序列
        
        参数:
            text: 输入文本
            add_special_tokens: 是否添加特殊标记（BOS/EOS）
            max_length: 最大序列长度（如果指定，将截断或填充到此长度）
            
        返回:
            ID序列
        """
        tokens = self.tokenize(text)
        ids = []
        
        # 添加开始标记
        if add_special_tokens:
            ids.append(self.bos_id)
            
        # 转换为ID
        for token in tokens:
            if token in self.token2id:
                ids.append(self.token2id[token])
            else:
                ids.append(self.unk_id)
                
        # 添加结束标记
        if add_special_tokens:
            ids.append(self.eos_id)
            
        # 处理最大长度
        if max_length is not None:
            if len(ids) > max_length:
                # 截断，确保保留EOS标记
                if add_special_tokens:
                    ids = ids[:max_length-1] + [self.eos_id]
                else:
                    ids = ids[:max_length]
            else:
                # 填充
                ids = ids + [self.pad_id] * (max_length - len(ids))
                
        return ids
    
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        将ID序列解码为文本
        
        参数:
            ids: ID序列
            skip_special_tokens: 是否跳过特殊标记
            
        返回:
            解码后的文本
        """
        special_ids = {self.pad_id, self.unk_id, self.bos_id, self.eos_id, self.mask_id} if skip_special_tokens else set()
        
        tokens = []
        for id in ids:
            if id in self.id2token and id not in special_ids:
                tokens.append(self.id2token[id])
                
        return ' '.join(tokens)
    
    def batch_encode(self, texts: List[str], add_special_tokens: bool = True, 
                    max_length: Optional[int] = None, padding: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        批量编码文本，并返回输入ID和注意力掩码
        
        参数:
            texts: 文本列表
            add_special_tokens: 是否添加特殊标记
            max_length: 最大序列长度
            padding: 是否填充到最大长度
            
        返回:
            (input_ids, attention_mask)，都是形状为[batch_size, seq_len]的张量
        """
        encoded = [self.encode(text, add_special_tokens, None) for text in texts]
        
        # 确定填充长度
        if max_length is None and padding:
            max_length = max(len(ids) for ids in encoded)
        elif max_length is None:
            max_length = max(len(ids) for ids in encoded)
            
        # 创建批次张量
        batch_size = len(texts)
        input_ids = torch.ones((batch_size, max_length), dtype=torch.long) * self.pad_id
        attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)
        
        # 填充序列
        for i, ids in enumerate(encoded):
            length = min(len(ids), max_length)
            input_ids[i, :length] = torch.tensor(ids[:length])
            attention_mask[i, :length] = 1
            
        return input_ids, attention_mask
    
    def save(self, path: str):
        """
        保存分词器
        
        参数:
            path: 保存路径
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 准备保存数据
        data = {
            'vocab_size': self.vocab_size,
            'token2id': self.token2id,
            'special_tokens': self.special_tokens
        }
        
        # 保存为JSON文件
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        logging.info(f"分词器已保存到 {path}")
        
    @classmethod
    def from_file(cls, path: str):
        """
        从文件加载分词器
        
        参数:
            path: 文件路径
            
        返回:
            Tokenizer实例
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # 创建分词器实例
        tokenizer = cls(vocab_size=data['vocab_size'], special_tokens=data['special_tokens'])
        
        # 重建词表
        tokenizer.token2id = data['token2id']
        tokenizer.id2token = {int(id): token for token, id in data['token2id'].items()}
        
        # 更新特殊标记ID
        tokenizer.pad_id = tokenizer.token2id.get(tokenizer.special_tokens['pad'], 0)
        tokenizer.unk_id = tokenizer.token2id.get(tokenizer.special_tokens['unk'], 1)
        tokenizer.bos_id = tokenizer.token2id.get(tokenizer.special_tokens['bos'], 2)
        tokenizer.eos_id = tokenizer.token2id.get(tokenizer.special_tokens['eos'], 3)
        tokenizer.mask_id = tokenizer.token2id.get(tokenizer.special_tokens['mask'], 4)
        
        logging.info(f"从 {path} 加载分词器，词表大小: {len(tokenizer.token2id)}")
        return tokenizer


class BPETokenizer(Tokenizer):
    """
    BPE(Byte-Pair Encoding)分词器实现
    
    BPE算法通过迭代合并最频繁的字符对来构建子词词表
    """
    def __init__(self, vocab_size=30000, special_tokens=None):
        super().__init__(vocab_size, special_tokens)
        self.merges = {}  # 存储合并规则
        
    def _get_stats(self, ids_list):
        """统计相邻符号对的频率"""
        counts = Counter()
        for ids in ids_list:
            for i in range(len(ids) - 1):
                pair = (ids[i], ids[i + 1])
                counts[pair] += 1
        return counts
    
    def _merge_pair(self, pair, ids_list):
        """合并指定的符号对"""
        first, second = pair
        new_token = first + second
        new_id = self._add_token(new_token)
        
        new_ids_list = []
        for ids in ids_list:
            i = 0
            new_ids = []
            while i < len(ids):
                if i < len(ids) - 1 and ids[i] == first and ids[i + 1] == second:
                    new_ids.append(new_id)
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1
            new_ids_list.append(new_ids)
        
        return new_ids_list
    
    def learn_bpe(self, texts, num_merges):
        """
        学习BPE合并规则
        
        参数:
            texts: 文本列表
            num_merges: 合并操作次数
        """
        # 初始化为字符级别的词表
        word_freqs = Counter()
        for text in texts:
            word_freqs.update(text.split())
        
        # 将单词分解为字符
        chars = set()
        word_ids = {}
        for word, freq in word_freqs.items():
            chars.update(word)
            word_ids[word] = [self._add_token(c) for c in word]
        
        # 存储单词及其出现频率
        ids_list = []
        for word, freq in word_freqs.items():
            ids_list.extend([word_ids[word]] * freq)
        
        # 执行指定次数的合并
        for i in range(num_merges):
            if len(self.token2id) >= self.vocab_size:
                break
                
            stats = self._get_stats(ids_list)
            if not stats:
                break
                
            # 找到最频繁的符号对
            best_pair = max(stats.items(), key=lambda x: x[1])[0]
            
            # 存储合并规则
            first, second = best_pair
            new_token = self.id2token[first] + self.id2token[second]
            self.merges[(first, second)] = self._add_token(new_token)
            
            # 应用合并规则
            ids_list = self._merge_pair(best_pair, ids_list)
            
            if (i + 1) % 100 == 0:
                logging.info(f"BPE合并进度: {i+1}/{num_merges}, 词表大小: {len(self.token2id)}")
    
    def tokenize(self, text):
        """使用BPE算法进行分词"""
        # 首先按空格分词
        words = text.split()
        tokens = []
        
        for word in words:
            # 如果单词已在词表中，直接添加
            if word in self.token2id:
                tokens.append(word)
                continue
                
            # 否则，将单词分解为字符
            chars = list(word)
            char_ids = [self.token2id.get(c, self.unk_id) for c in chars]
            
            # 应用合并规则
            while len(char_ids) > 1:
                pairs = [(char_ids[i], char_ids[i+1]) for i in range(len(char_ids)-1)]
                # 找到可以合并的最高优先级对
                mergeable_pairs = [p for p in pairs if p in self.merges]
                
                if not mergeable_pairs:
                    break
                    
                # 执行合并
                pair_to_merge = mergeable_pairs[0]  # 简化，实际上应该按优先级排序
                idx = pairs.index(pair_to_merge)
                
                # 更新char_ids
                char_ids = (char_ids[:idx] + 
                           [self.merges[pair_to_merge]] + 
                           char_ids[idx+2:])
            
            # 将ID转换回token
            word_tokens = [self.id2token.get(id, self.special_tokens['unk']) for id in char_ids]
            tokens.extend(word_tokens)
            
        return tokens
    
    def save(self, path):
        """保存BPE分词器"""
        data = {
            'vocab_size': self.vocab_size,
            'token2id': self.token2id,
            'special_tokens': self.special_tokens,
            'merges': {f"{k[0]},{k[1]}": v for k, v in self.merges.items()}
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        logging.info(f"BPE分词器已保存到 {path}")
    
    @classmethod
    def from_file(cls, path):
        """从文件加载BPE分词器"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        tokenizer = cls(vocab_size=data['vocab_size'], special_tokens=data['special_tokens'])
        
        tokenizer.token2id = {k: int(v) for k, v in data['token2id'].items()}
        tokenizer.id2token = {int(id): token for token, id in tokenizer.token2id.items()}
        
        # 重建合并规则
        tokenizer.merges = {}
        for pair_str, id in data['merges'].items():
            first, second = pair_str.split(',')
            tokenizer.merges[(int(first), int(second))] = int(id)
            
        # 更新特殊标记ID
        tokenizer.pad_id = tokenizer.token2id.get(tokenizer.special_tokens['pad'], 0)
        tokenizer.unk_id = tokenizer.token2id.get(tokenizer.special_tokens['unk'], 1)
        tokenizer.bos_id = tokenizer.token2id.get(tokenizer.special_tokens['bos'], 2)
        tokenizer.eos_id = tokenizer.token2id.get(tokenizer.special_tokens['eos'], 3)
        tokenizer.mask_id = tokenizer.token2id.get(tokenizer.special_tokens['mask'], 4)
        
        logging.info(f"从 {path} 加载BPE分词器，词表大小: {len(tokenizer.token2id)}")
        return tokenizer
