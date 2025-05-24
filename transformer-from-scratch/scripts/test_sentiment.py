#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试微调后的情感分析模型
"""

import torch
import argparse
import logging
import os
import sys
import json

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.transformer import TransformerLM, create_look_ahead_mask
from data.tokenizer import Tokenizer
from scripts.finetune import TransformerForClassification

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def load_model(model_path, tokenizer, num_classes=2):
    """
    加载微调后的模型
    
    参数:
        model_path: 模型路径
        tokenizer: 分词器
        num_classes: 类别数
        
    返回:
        微调后的模型
    """
    # 加载检查点
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # 创建基础模型
    base_model = TransformerLM(
        vocab_size=len(tokenizer.token2id),
        d_model=128,
        n_layers=2,
        n_heads=2,
        d_ff=512,
        max_seq_len=128
    )
    
    # 创建分类模型
    model = TransformerForClassification(
        transformer=base_model,
        num_classes=num_classes,
        dropout=0.1
    )
    
    # 加载模型参数
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # 设置为评估模式
    
    logging.info(f"从 {model_path} 加载微调模型")
    return model

def predict_sentiment(text, model, tokenizer, device):
    """
    预测文本的情感
    
    参数:
        text: 输入文本
        model: 模型
        tokenizer: 分词器
        device: 设备
        
    返回:
        预测的情感类别 (0表示负面, 1表示正面)
        预测的概率
    """
    # 对文本进行分词
    tokens = tokenizer.encode(text)
    input_ids = torch.tensor([tokens], device=device)
    
    # 创建注意力掩码
    attention_mask = torch.ones_like(input_ids)
    
    # 进行预测
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=1)
        prediction = torch.argmax(probs, dim=1).item()
        confidence = probs[0][prediction].item()
    
    return prediction, confidence

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="测试情感分析模型")
    parser.add_argument("--model", type=str, default="saved_models/finetune/model_final.pt", help="模型路径")
    parser.add_argument("--tokenizer", type=str, default="data/tokenizer/tokenizer_small.json", help="分词器路径")
    parser.add_argument("--text", type=str, required=True, help="要分析的文本")
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"使用设备: {device}")
    
    # 加载分词器
    tokenizer = Tokenizer.from_file(args.tokenizer)
    logging.info(f"从 {args.tokenizer} 加载分词器，词表大小: {len(tokenizer.token2id)}")
    
    # 加载模型
    model = load_model(args.model, tokenizer)
    model.to(device)
    
    # 进行预测
    text = args.text
    logging.info(f"输入文本: '{text}'")
    
    prediction, confidence = predict_sentiment(text, model, tokenizer, device)
    sentiment = "正面" if prediction == 1 else "负面"
    
    logging.info(f"预测情感: {sentiment} (置信度: {confidence:.4f})")
    
    # 返回结果
    result = {
        "text": text,
        "sentiment": sentiment,
        "prediction": int(prediction),
        "confidence": float(confidence)
    }
    
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
