import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import SelfAttention


class PositionwiseFeedForward(nn.Module):
    """
    位置前馈网络
    
    FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        参数:
            d_model: 输入和输出的维度
            d_ff: 隐藏层维度
            dropout: dropout比率
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        """
        参数:
            x: [batch_size, seq_len, d_model]
        
        返回:
            [batch_size, seq_len, d_model]
        """
        # 保存输入用于残差连接
        residual = x
        
        # 前馈网络
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        
        # 残差连接和层归一化
        x = self.layer_norm(x + residual)
        
        return x


class EncoderLayer(nn.Module):
    """
    Transformer编码器层
    
    由自注意力层和位置前馈网络组成
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        """
        参数:
            d_model: 模型维度
            n_heads: 注意力头数
            d_ff: 前馈网络隐藏层维度
            dropout: dropout比率
        """
        super().__init__()
        self.self_attention = SelfAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
    def forward(self, x, mask=None):
        """
        参数:
            x: [batch_size, seq_len, d_model]
            mask: [batch_size, 1, seq_len, seq_len]
        
        返回:
            输出: [batch_size, seq_len, d_model]
            注意力权重: [batch_size, n_heads, seq_len, seq_len]
        """
        # 自注意力层
        attn_output, attention_weights = self.self_attention(x, mask)
        
        # 位置前馈网络
        output = self.feed_forward(attn_output)
        
        return output, attention_weights


class Encoder(nn.Module):
    """
    Transformer编码器
    
    由多个编码器层堆叠而成
    """
    def __init__(self, n_layers, d_model, n_heads, d_ff, dropout=0.1):
        """
        参数:
            n_layers: 编码器层数
            d_model: 模型维度
            n_heads: 注意力头数
            d_ff: 前馈网络隐藏层维度
            dropout: dropout比率
        """
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
    def forward(self, x, mask=None):
        """
        参数:
            x: [batch_size, seq_len, d_model]
            mask: [batch_size, 1, seq_len, seq_len]
        
        返回:
            输出: [batch_size, seq_len, d_model]
            所有层的注意力权重列表
        """
        attention_weights = []
        
        for layer in self.layers:
            x, attn_weights = layer(x, mask)
            attention_weights.append(attn_weights)
            
        return x, attention_weights
