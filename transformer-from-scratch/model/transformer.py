import torch
import torch.nn as nn
import torch.nn.functional as F

from .embeddings import TransformerEmbedding
from .encoder import Encoder
from .decoder import Decoder, create_padding_mask, create_look_ahead_mask


class Transformer(nn.Module):
    """
    完整的Transformer模型
    
    包含编码器和解码器
    """
    def __init__(self, 
                 src_vocab_size, 
                 tgt_vocab_size, 
                 d_model=512, 
                 n_layers=6, 
                 n_heads=8, 
                 d_ff=2048, 
                 max_seq_len=5000, 
                 dropout=0.1):
        """
        参数:
            src_vocab_size: 源语言词表大小
            tgt_vocab_size: 目标语言词表大小
            d_model: 模型维度
            n_layers: 编码器和解码器层数
            n_heads: 注意力头数
            d_ff: 前馈网络隐藏层维度
            max_seq_len: 最大序列长度
            dropout: dropout比率
        """
        super().__init__()
        
        # 源语言和目标语言的嵌入层
        self.src_embedding = TransformerEmbedding(src_vocab_size, d_model, max_seq_len, dropout)
        self.tgt_embedding = TransformerEmbedding(tgt_vocab_size, d_model, max_seq_len, dropout)
        
        # 编码器和解码器
        self.encoder = Encoder(n_layers, d_model, n_heads, d_ff, dropout)
        self.decoder = Decoder(n_layers, d_model, n_heads, d_ff, dropout)
        
        # 输出线性层和softmax
        self.final_layer = nn.Linear(d_model, tgt_vocab_size)
        
        # 初始化参数
        self._init_parameters()
        
    def _init_parameters(self):
        """
        初始化模型参数
        
        使用Xavier均匀分布初始化线性层权重
        将所有偏置初始化为零
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
    def create_masks(self, src, tgt, pad_idx=0):
        """
        创建编码器和解码器的掩码
        
        参数:
            src: [batch_size, src_seq_len] 源序列
            tgt: [batch_size, tgt_seq_len] 目标序列
            pad_idx: 填充标记的索引值
            
        返回:
            src_mask: [batch_size, 1, 1, src_seq_len] 源序列填充掩码
            tgt_mask: [batch_size, 1, tgt_seq_len, tgt_seq_len] 目标序列掩码
        """
        # 源序列填充掩码
        src_mask = create_padding_mask(src, pad_idx)
        
        # 目标序列掩码（填充掩码和前瞻掩码的组合）
        tgt_padding_mask = create_padding_mask(tgt, pad_idx)
        tgt_look_ahead_mask = create_look_ahead_mask(tgt.size(1)).to(tgt.device)
        
        # 组合两个掩码，确保tgt_padding_mask的维度与tgt_look_ahead_mask匹配
        tgt_padding_mask = tgt_padding_mask.expand(-1, -1, tgt.size(1), -1)
        tgt_mask = tgt_padding_mask & tgt_look_ahead_mask.unsqueeze(0)
        
        return src_mask, tgt_mask
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        Transformer前向传播
        
        参数:
            src: [batch_size, src_seq_len] 源序列
            tgt: [batch_size, tgt_seq_len] 目标序列
            src_mask: [batch_size, 1, 1, src_seq_len] 源序列掩码
            tgt_mask: [batch_size, 1, tgt_seq_len, tgt_seq_len] 目标序列掩码
            
        返回:
            输出概率: [batch_size, tgt_seq_len, tgt_vocab_size]
            编码器注意力权重
            解码器自注意力权重
            解码器交叉注意力权重
        """
        # 如果没有提供掩码，自动创建
        if src_mask is None or tgt_mask is None:
            src_mask, tgt_mask = self.create_masks(src, tgt)
            
        # 编码器处理
        src_embedded = self.src_embedding(src)
        encoder_output, encoder_attention = self.encoder(src_embedded, src_mask)
        
        # 解码器处理（预测时，tgt是已生成的目标序列）
        tgt_embedded = self.tgt_embedding(tgt)
        decoder_output, decoder_self_attention, decoder_cross_attention = self.decoder(
            tgt_embedded, encoder_output, src_mask, tgt_mask)
        
        # 最终输出层
        output = self.final_layer(decoder_output)
        
        return output, encoder_attention, decoder_self_attention, decoder_cross_attention
    
    def encode(self, src, src_mask=None):
        """
        只执行编码器部分
        
        参数:
            src: [batch_size, src_seq_len] 源序列
            src_mask: [batch_size, 1, 1, src_seq_len] 源序列掩码
            
        返回:
            编码器输出: [batch_size, src_seq_len, d_model]
        """
        if src_mask is None:
            src_mask = create_padding_mask(src)
            
        src_embedded = self.src_embedding(src)
        encoder_output, _ = self.encoder(src_embedded, src_mask)
        
        return encoder_output, src_mask
    
    def decode(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        """
        只执行解码器部分
        
        参数:
            tgt: [batch_size, tgt_seq_len] 目标序列
            encoder_output: [batch_size, src_seq_len, d_model] 编码器输出
            src_mask: [batch_size, 1, 1, src_seq_len] 源序列掩码
            tgt_mask: [batch_size, 1, tgt_seq_len, tgt_seq_len] 目标序列掩码
            
        返回:
            输出概率: [batch_size, tgt_seq_len, tgt_vocab_size]
        """
        if tgt_mask is None:
            # 只创建前瞻掩码（用于自回归生成）
            tgt_mask = create_look_ahead_mask(tgt.size(1)).to(tgt.device).unsqueeze(0).unsqueeze(1)
            
        tgt_embedded = self.tgt_embedding(tgt)
        decoder_output, _, _ = self.decoder(tgt_embedded, encoder_output, src_mask, tgt_mask)
        output = self.final_layer(decoder_output)
        
        return output
    
    def generate(self, src, max_len, start_symbol, end_symbol=None, temperature=1.0, top_k=0, top_p=0.0):
        """
        使用模型生成序列
        
        参数:
            src: [batch_size, src_seq_len] 源序列
            max_len: 生成的最大长度
            start_symbol: 开始符号的索引
            end_symbol: 结束符号的索引（可选）
            temperature: 采样温度，控制分布的峰值
            top_k: 仅考虑概率最高的k个token（如果为0，则不使用）
            top_p: 累积概率阈值，仅考虑概率和达到p的token（如果为0，则不使用）
            
        返回:
            生成的序列: [batch_size, seq_len]
        """
        batch_size = src.size(0)
        device = src.device
        
        # 编码源序列
        encoder_output, src_mask = self.encode(src)
        
        # 初始化目标序列为开始符号
        tgt = torch.ones(batch_size, 1).fill_(start_symbol).long().to(device)
        
        for i in range(max_len - 1):
            # 解码当前目标序列
            output = self.decode(tgt, encoder_output, src_mask)
            
            # 获取最后一个时间步的预测
            output = output[:, -1, :] / temperature
            
            # 应用top-k采样（如果指定）
            if top_k > 0:
                indices_to_remove = output < torch.topk(output, top_k)[0][..., -1, None]
                output[indices_to_remove] = -float('inf')
            
            # 应用top-p采样（如果指定）
            if top_p > 0.0:
                sorted_logits, sorted_indices = torch.sort(output, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # 移除概率累积超过阈值的token
                sorted_indices_to_remove = cumulative_probs > top_p
                # 保留第一个token
                sorted_indices_to_remove[..., 0] = 0
                
                # 将排序后的索引恢复到原始顺序
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove)
                output[indices_to_remove] = -float('inf')
            
            # 对概率分布进行采样
            probabilities = F.softmax(output, dim=-1)
            next_token = torch.multinomial(probabilities, 1)
            
            # 将新token添加到目标序列
            tgt = torch.cat([tgt, next_token], dim=1)
            
            # 如果所有序列都生成了结束符号，则提前停止
            if end_symbol is not None and (next_token == end_symbol).all():
                break
        
        return tgt


class TransformerLM(nn.Module):
    """
    Transformer语言模型（仅使用解码器）
    
    用于预训练和生成任务
    """
    def __init__(self, 
                 vocab_size, 
                 d_model=512, 
                 n_layers=6, 
                 n_heads=8, 
                 d_ff=2048, 
                 max_seq_len=5000, 
                 dropout=0.1):
        """
        参数:
            vocab_size: 词表大小
            d_model: 模型维度
            n_layers: 解码器层数
            n_heads: 注意力头数
            d_ff: 前馈网络隐藏层维度
            max_seq_len: 最大序列长度
            dropout: dropout比率
        """
        super().__init__()
        
        # 嵌入层
        self.embedding = TransformerEmbedding(vocab_size, d_model, max_seq_len, dropout)
        
        # 使用解码器层，但去掉交叉注意力（因为没有编码器输出）
        self.decoder = Encoder(n_layers, d_model, n_heads, d_ff, dropout)
        
        # 输出线性层和softmax
        self.final_layer = nn.Linear(d_model, vocab_size)
        
        # 初始化参数
        self._init_parameters()
        
    def _init_parameters(self):
        """初始化模型参数"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def forward(self, x, mask=None):
        """
        TransformerLM前向传播
        
        参数:
            x: [batch_size, seq_len] 输入序列
            mask: [batch_size, 1, seq_len, seq_len] 掩码
            
        返回:
            输出概率: [batch_size, seq_len, vocab_size]
        """
        # 如果没有提供掩码，自动创建
        if mask is None:
            # 创建前瞻掩码（用于自回归生成）
            seq_len = x.size(1)
            mask = create_look_ahead_mask(seq_len).to(x.device)
            # 调整掩码形状为 [batch_size, 1, seq_len, seq_len]
            # 这样确保它能在attention计算中被正确广播
            mask = mask.unsqueeze(0).expand(x.size(0), -1, -1)
            
        # 嵌入层
        x = self.embedding(x)
        
        # 解码器层
        x, _ = self.decoder(x, mask)
        
        # 最终输出层
        output = self.final_layer(x)
        
        return output
    
    def generate(self, prompt, max_len, temperature=1.0, top_k=0, top_p=0.0, eos_token_id=None):
        """
        生成文本
        
        参数:
            prompt: [batch_size, prompt_len] 提示文本
            max_len: 生成的最大长度
            temperature: 采样温度
            top_k: top-k采样参数
            top_p: top-p采样参数
            eos_token_id: 结束符号ID
            
        返回:
            生成的文本: [batch_size, seq_len]
        """
        batch_size = prompt.size(0)
        device = prompt.device
        
        # 初始化输出为提示文本
        generated = prompt.clone()
        
        # 循环生成每个新token
        for _ in range(max_len):
            # 为当前序列创建掩码
            seq_len = generated.size(1)
            mask = create_look_ahead_mask(seq_len).to(device)
            mask = mask.unsqueeze(0).unsqueeze(1)
            
            # 前向传播获取下一个token的概率
            output = self.forward(generated, mask)
            
            # 获取最后一个位置的输出
            next_token_logits = output[:, -1, :] / temperature
            
            # 应用top-k采样
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float('inf')
            
            # 应用top-p采样
            if top_p > 0.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = -float('inf')
            
            # 计算概率分布并采样
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # 添加采样的token到序列中
            generated = torch.cat((generated, next_token), dim=1)
            
            # 如果生成了结束符号，则停止生成
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
                
        return generated
