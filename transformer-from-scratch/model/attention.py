import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ScaledDotProductAttention(nn.Module):
    """
    计算缩放点积注意力
    
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
    """
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        """
        参数:
            query: [batch_size, n_heads, seq_len_q, d_k]
            key: [batch_size, n_heads, seq_len_k, d_k]
            value: [batch_size, n_heads, seq_len_v, d_v] (通常 seq_len_k == seq_len_v)
            mask: [batch_size, 1, 1, seq_len_k] 或 [batch_size, 1, seq_len_q, seq_len_k]
        
        返回:
            注意力输出: [batch_size, n_heads, seq_len_q, d_v]
            注意力权重: [batch_size, n_heads, seq_len_q, seq_len_k]
        """
        d_k = query.size(-1)
        
        # 计算注意力分数
        # [batch_size, n_heads, seq_len_q, d_k] x [batch_size, n_heads, d_k, seq_len_k]
        # -> [batch_size, n_heads, seq_len_q, seq_len_k]
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        # 应用掩码（可选）
        if mask is not None:
            # 确保掩码维度与scores匹配
            # 如果mask是[batch_size, 1, 1, seq_len_k]或[batch_size, 1, seq_len_q, seq_len_k]
            # 我们需要确保它能正确广播到scores的形状[batch_size, n_heads, seq_len_q, seq_len_k]
            if mask.dim() == 3:
                # 如果掩码是3维的，将其扩展为4维 [batch_size, 1, seq_len_q, seq_len_k]
                mask = mask.unsqueeze(1)
            elif mask.dim() == 2:
                # 如果掩码是2维的，将其扩展为4维 [batch_size, 1, 1, seq_len_k]
                mask = mask.unsqueeze(1).unsqueeze(1)
            
            # 填充非常小的值，使得softmax后接近0
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 注意力权重: [batch_size, n_heads, seq_len_q, seq_len_k]
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 输出: [batch_size, n_heads, seq_len_q, d_v]
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制
    
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W^O
    where head_i = Attention(Q * W_i^Q, K * W_i^K, V * W_i^V)
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        
        # 确保模型维度可以被头数整除
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # 每个头的维度
        self.d_v = d_model // n_heads
        
        # 为每个头创建线性变换
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention(dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def split_heads(self, x, batch_size):
        """
        将输入拆分为多个头
        
        从 [batch_size, seq_len, d_model] 拆分为 [batch_size, n_heads, seq_len, d_k]
        """
        x = x.view(batch_size, -1, self.n_heads, self.d_k)
        return x.permute(0, 2, 1, 3)
    
    def combine_heads(self, x, batch_size):
        """
        将多个头合并
        
        从 [batch_size, n_heads, seq_len, d_v] 合并为 [batch_size, seq_len, d_model]
        """
        x = x.permute(0, 2, 1, 3)
        return x.contiguous().view(batch_size, -1, self.d_model)
    
    def forward(self, query, key, value, mask=None, residual=None):
        """
        多头注意力前向传播
        
        参数:
            query: [batch_size, seq_len_q, d_model]
            key: [batch_size, seq_len_k, d_model]
            value: [batch_size, seq_len_v, d_model]
            mask: [batch_size, 1, seq_len_q, seq_len_k]
            residual: 用于残差连接的输入（可选，默认使用query）
        
        返回:
            输出: [batch_size, seq_len_q, d_model]
            注意力权重: [batch_size, n_heads, seq_len_q, seq_len_k]
        """
        batch_size = query.size(0)
        
        # 如果没有提供残差连接输入，使用query
        if residual is None:
            residual = query
            
        # 线性投影
        q = self.W_q(query)  # [batch_size, seq_len_q, d_model]
        k = self.W_k(key)    # [batch_size, seq_len_k, d_model]
        v = self.W_v(value)  # [batch_size, seq_len_v, d_model]
        
        # 拆分多头
        q = self.split_heads(q, batch_size)  # [batch_size, n_heads, seq_len_q, d_k]
        k = self.split_heads(k, batch_size)  # [batch_size, n_heads, seq_len_k, d_k]
        v = self.split_heads(v, batch_size)  # [batch_size, n_heads, seq_len_v, d_v]
        
        # 应用缩放点积注意力
        attn_output, attention_weights = self.attention(q, k, v, mask)
        
        # 合并多头
        output = self.combine_heads(attn_output, batch_size)  # [batch_size, seq_len_q, d_model]
        
        # 最终线性投影
        output = self.W_o(output)  # [batch_size, seq_len_q, d_model]
        
        # 残差连接和层归一化
        output = self.layer_norm(output + residual)
        
        return output, attention_weights


class SelfAttention(nn.Module):
    """
    自注意力模块，用于处理序列内部的关系
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(d_model, n_heads, dropout)
        
    def forward(self, x, mask=None):
        """
        自注意力前向传播
        
        参数:
            x: [batch_size, seq_len, d_model]
            mask: [batch_size, 1, seq_len, seq_len]
        
        返回:
            输出: [batch_size, seq_len, d_model]
            注意力权重: [batch_size, n_heads, seq_len, seq_len]
        """
        return self.multi_head_attention(x, x, x, mask)


class CrossAttention(nn.Module):
    """
    交叉注意力模块，用于解码器中处理编码器输出与解码器输入的关系
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(d_model, n_heads, dropout)
        
    def forward(self, x, encoder_output, mask=None):
        """
        交叉注意力前向传播
        
        参数:
            x: [batch_size, seq_len_q, d_model] 解码器输入
            encoder_output: [batch_size, seq_len_k, d_model] 编码器输出
            mask: [batch_size, 1, seq_len_q, seq_len_k]
        
        返回:
            输出: [batch_size, seq_len_q, d_model]
            注意力权重: [batch_size, n_heads, seq_len_q, seq_len_k]
        """
        return self.multi_head_attention(x, encoder_output, encoder_output, mask, residual=x)
