import torch
import torch.nn as nn
import math


class TokenEmbedding(nn.Module):
    """
    将输入的token索引转换为向量表示
    """
    def __init__(self, vocab_size, d_model):
        """
        参数:
            vocab_size: 词表大小
            d_model: 嵌入向量维度
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        
    def forward(self, x):
        """
        参数:
            x: [batch_size, seq_len] token索引
        
        返回:
            [batch_size, seq_len, d_model] 嵌入向量
        """
        # 安全检查: 确保所有token ID在词表范围内
        vocab_size = self.embedding.num_embeddings
        max_id = x.max().item()
        
        if max_id >= vocab_size:
            # 发现超出范围的ID，限制在词表范围内
            print(f"\n[WARNING] 发现超出范围的token ID: {max_id} >= {vocab_size}")
            print(f"\n[WARNING] 自动将超出范围的ID限制在词表大小范围内")
            # 限制范围在[0, vocab_size-1]            
            x = torch.clamp(x, 0, vocab_size-1)
        
        # 对嵌入向量乘以sqrt(d_model)
        # 这有助于梯度在反向传播时更稳定
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    位置编码模块，为序列中的每个位置添加位置信息
    
    使用正弦和余弦函数的固定位置编码:
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        """
        参数:
            d_model: 嵌入向量维度
            max_len: 支持的最大序列长度
            dropout: dropout比率
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 应用正弦函数到偶数索引
        pe[:, 0::2] = torch.sin(position * div_term)
        # 应用余弦函数到奇数索引
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 添加批次维度
        pe = pe.unsqueeze(0)
        
        # 注册为缓冲区，这样在保存模型时会被保存，但不是模型参数
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        参数:
            x: [batch_size, seq_len, d_model] 输入嵌入向量
        
        返回:
            [batch_size, seq_len, d_model] 添加位置编码后的向量
        """
        # 为输入添加位置编码
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEmbedding(nn.Module):
    """
    Transformer的完整嵌入层，包括token嵌入和位置编码
    """
    def __init__(self, vocab_size, d_model, max_len=5000, dropout=0.1):
        """
        参数:
            vocab_size: 词表大小
            d_model: 嵌入向量维度
            max_len: 支持的最大序列长度
            dropout: dropout比率
        """
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)
        
    def forward(self, x):
        """
        参数:
            x: [batch_size, seq_len] token索引
        
        返回:
            [batch_size, seq_len, d_model] 嵌入向量
        """
        # 先进行token嵌入，再添加位置编码
        return self.positional_encoding(self.token_embedding(x))
