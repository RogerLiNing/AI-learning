import os
import sys
import torch
import argparse
import logging

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.transformer import TransformerLM
from data.tokenizer import Tokenizer, BPETokenizer

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_model(model_path, tokenizer, device='cpu'):
    """加载预训练模型"""
    # 加载检查点
    checkpoint = torch.load(model_path, map_location=device)
    
    # 创建模型
    model = TransformerLM(
        vocab_size=len(tokenizer.token2id),
        d_model=128,  # 应与预训练配置一致
        n_layers=2,   # 应与预训练配置一致
        n_heads=2,    # 应与预训练配置一致
        d_ff=512,     # 应与预训练配置一致
        max_seq_len=128  # 应与预训练配置一致
    )
    
    # 加载模型参数
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 将模型移至设备并设置为评估模式
    model.to(device)
    model.eval()
    
    logging.info(f"从 {model_path} 加载模型")
    return model

def generate_text(model, tokenizer, prompt, max_length=50, temperature=1.0, top_p=0.9, device='cpu'):
    """生成文本"""
    # 编码提示文本
    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    input_ids = torch.tensor([input_ids]).to(device)
    
    # 设置模型为评估模式
    model.eval()
    
    with torch.no_grad():
        # 通过模型的generate方法生成文本
        output_ids = model.generate(
            input_ids,
            max_len=max_length,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=tokenizer.eos_id
        )
        
        # 解码生成的文本
        generated_text = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)
        
    return generated_text

def main():
    parser = argparse.ArgumentParser(description="测试预训练的Transformer模型")
    parser.add_argument("--model", type=str, default="saved_models/pretrain/model_final.pt", help="模型路径")
    parser.add_argument("--tokenizer", type=str, default="data/tokenizer/tokenizer_small.json", help="分词器路径")
    parser.add_argument("--prompt", type=str, default="人工智能", help="提示文本")
    parser.add_argument("--max_length", type=int, default=30, help="生成的最大长度")
    parser.add_argument("--temperature", type=float, default=0.7, help="采样温度")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p采样参数")
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"模型文件不存在: {args.model}")
    if not os.path.exists(args.tokenizer):
        raise FileNotFoundError(f"分词器文件不存在: {args.tokenizer}")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"使用设备: {device}")
    
    # 加载分词器
    if "bpe" in args.tokenizer:
        tokenizer = BPETokenizer.from_file(args.tokenizer)
    else:
        tokenizer = Tokenizer.from_file(args.tokenizer)
    logging.info(f"从 {args.tokenizer} 加载分词器")
    
    # 加载模型
    model = load_model(args.model, tokenizer, device)
    
    # 生成文本
    logging.info(f"使用提示文本: '{args.prompt}'")
    generated_text = generate_text(
        model, 
        tokenizer, 
        args.prompt, 
        max_length=args.max_length,
        temperature=args.temperature,
        top_p=args.top_p,
        device=device
    )
    
    logging.info(f"生成的文本: {generated_text}")
    
if __name__ == "__main__":
    main()
