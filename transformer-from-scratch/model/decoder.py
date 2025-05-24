import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import SelfAttention, CrossAttention
from .encoder import PositionwiseFeedForward


class DecoderLayer(nn.Module):
    """
    Transformer解码器层
    
    包含三个子层:
    1. 掩码自注意力层
    2. 编码器-解码器交叉注意力层
    3. 位置前馈网络
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
        self.masked_self_attention = SelfAttention(d_model, n_heads, dropout)
        self.cross_attention = CrossAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        参数:
            x: [batch_size, tgt_seq_len, d_model]
            encoder_output: [batch_size, src_seq_len, d_model]
            src_mask: [batch_size, 1, tgt_seq_len, src_seq_len]
            tgt_mask: [batch_size, 1, tgt_seq_len, tgt_seq_len]
        
        返回:
            输出: [batch_size, tgt_seq_len, d_model]
            自注意力权重: [batch_size, n_heads, tgt_seq_len, tgt_seq_len]
            交叉注意力权重: [batch_size, n_heads, tgt_seq_len, src_seq_len]
        """
        # 掩码自注意力层
        self_attn_output, self_attn_weights = self.masked_self_attention(x, tgt_mask)
        
        # 编码器-解码器交叉注意力层
        cross_attn_output, cross_attn_weights = self.cross_attention(
            self_attn_output, encoder_output, src_mask)
        
        # 位置前馈网络
        output = self.feed_forward(cross_attn_output)
        
        return output, self_attn_weights, cross_attn_weights


class Decoder(nn.Module):
    """
    Transformer解码器
    
    由多个解码器层堆叠而成
    """
    def __init__(self, n_layers, d_model, n_heads, d_ff, dropout=0.1):
        """
        参数:
            n_layers: 解码器层数
            d_model: 模型维度
            n_heads: 注意力头数
            d_ff: 前馈网络隐藏层维度
            dropout: dropout比率
        """
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        参数:
            x: [batch_size, tgt_seq_len, d_model]
            encoder_output: [batch_size, src_seq_len, d_model]
            src_mask: [batch_size, 1, tgt_seq_len, src_seq_len]
            tgt_mask: [batch_size, 1, tgt_seq_len, tgt_seq_len]
        
        返回:
            输出: [batch_size, tgt_seq_len, d_model]
            自注意力权重列表
            交叉注意力权重列表
        """
        self_attention_weights = []
        cross_attention_weights = []
        
        for layer in self.layers:
            x, self_attn, cross_attn = layer(x, encoder_output, src_mask, tgt_mask)
            self_attention_weights.append(self_attn)
            cross_attention_weights.append(cross_attn)
            
        return x, self_attention_weights, cross_attention_weights


def create_look_ahead_mask(size):
    """
    创建前瞻掩码（上三角矩阵）
    
    参数:
        size: 序列长度
        
    返回:
        [size, size] 掩码矩阵
    """
    # 创建上三角矩阵（包括对角线）
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
    # 将True变为0，False变为1
    return mask.logical_not()


def create_padding_mask(seq, pad_idx=0):
    """
    创建填充掩码
    
    参数:
        seq: [batch_size, seq_len] 输入序列
        pad_idx: 填充标记的索引值
        
    返回:
        [batch_size, 1, 1, seq_len] 掩码
    """
    # 创建掩码，将pad_idx位置标记为True
    mask = (seq == pad_idx).bool()
    # 扩展维度以适合注意力机制
    return mask.unsqueeze(1).unsqueeze(2)
