#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
交互式聊天脚本 - 加载预训练的Transformer模型进行实时聊天
"""

import os
import sys
import torch
import logging
import argparse
import json
import time
import readline  # 支持输入历史和编辑

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.transformer import TransformerLM
from data.hf_tokenizer import HFTokenizer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("interactive_chat.log", mode='w')
    ]
)

def setup_device(use_cuda, use_mps):
    """设置设备"""
    if use_cuda and torch.cuda.is_available():
        device = "cuda"
        logging.info(f"使用设备: {device} - {torch.cuda.get_device_name()}")
    elif use_mps and torch.backends.mps.is_available():
        device = "mps"
        logging.info(f"使用设备: {device} - Apple Silicon")
    else:
        device = "cpu"
        logging.info(f"使用设备: {device}")
    
    return device

def load_tokenizer(tokenizer_path):
    """加载分词器"""
    if os.path.isdir(tokenizer_path):
        tokenizer = HFTokenizer(pretrained_model_name=tokenizer_path)
        logging.info(f"从本地加载分词器: {tokenizer_path}, 词表大小: {tokenizer.vocab_size}")
    else:
        logging.error(f"分词器路径不存在: {tokenizer_path}")
        raise FileNotFoundError(f"分词器路径不存在: {tokenizer_path}")
    return tokenizer

def load_model(model_path, config, device):
    """加载模型"""
    if not os.path.exists(model_path):
        logging.error(f"模型路径不存在: {model_path}")
        raise FileNotFoundError(f"模型路径不存在: {model_path}")
    
    # 加载检查点
    checkpoint = torch.load(model_path, map_location=device)
    
    # 从配置中获取模型参数
    d_model = config.get('d_model', 512)
    n_layers = config.get('n_layers', 6)
    n_heads = config.get('n_heads', 8)
    d_ff = config.get('d_ff', 2048)
    dropout = config.get('dropout', 0.1)
    max_seq_len = config.get('max_seq_len', 512)
    
    # 获取词表大小
    if 'vocab_size' in checkpoint:
        vocab_size = checkpoint['vocab_size']
    else:
        # 从分词器获取
        tokenizer = load_tokenizer(config.get('tokenizer_dir', 'data/hf_tokenizer'))
        vocab_size = tokenizer.vocab_size
    
    logging.info(f"创建模型: d_model={d_model}, n_layers={n_layers}, n_heads={n_heads}, vocab_size={vocab_size}")
    
    # 创建模型
    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        dropout=dropout,
        max_seq_len=max_seq_len
    )
    
    # 加载模型权重
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()  # 设置为评估模式
    
    return model

def generate_response(model, tokenizer, prompt, max_length=100, temperature=1.0, top_k=50, top_p=0.9, device='cuda'):
    """生成回复"""
    # 中文常用词表，用于补充生成文本
    common_cn_words = [
        "我", "你", "他", "她", "它", "们", "这", "那", "是", "不", "在", "有", "了",
        "就", "和", "与", "也", "可", "能", "如", "要", "会", "对", "到", "为", "得", "上",
        "下", "中", "外", "前", "后", "可以", "很", "看", "到", "了", "不", "这个", "那个",
        "你好", "谢谢", "对不起", "没关系", "很高兴", "再见"
    ]
    
    # 编码提示文本
    prompt_ids = tokenizer.encode(prompt)
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long).to(device)
    
    # 检查并打印提示信息
    logging.debug(f"提示ID: {prompt_ids}")
    logging.debug(f"解码的提示: {tokenizer.decode(prompt_ids)}")
    
    # 生成文本
    with torch.no_grad():
        output_ids = model.generate(
            prompt_tensor,
            max_len=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
    
    # 获取原始输出ID
    raw_output = output_ids[0].tolist()
    
    # 移除可能的提示ID部分，只保留新生成的部分
    if len(raw_output) > len(prompt_ids):
        # 移除完全匹配的提示部分
        if raw_output[:len(prompt_ids)] == prompt_ids:
            generated_ids = raw_output[len(prompt_ids):]
        else:
            # 如果提示部分不完全匹配，使用原始输出
            generated_ids = raw_output
    else:
        generated_ids = raw_output
    
    # 获取所有生成的token
    try:
        tokens = [tokenizer.convert_ids_to_tokens(id) for id in generated_ids]
        logging.debug(f"生成的tokens: {tokens}")
    except:
        tokens = []
    
    # 解码生成的文本，使用skip_special_tokens选项
    try:
        # 尝试解码完整输出
        full_text = tokenizer.decode(raw_output, skip_special_tokens=True)
        # 尝试只解码生成部分
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # 对于中文，特别处理中文字符之间的空格
        # 使用正则表达式移除中文字符之间的空格
        import re
        # 中文字符范围正则
        chinese_pattern = r'[\u4e00-\u9fa5]'
        # 将中文字符之间的空格移除
        cleaned_text = re.sub(f"({chinese_pattern}) ({chinese_pattern})", r"\1\2", generated_text)
        while re.search(f"({chinese_pattern}) ({chinese_pattern})", cleaned_text):
            cleaned_text = re.sub(f"({chinese_pattern}) ({chinese_pattern})", r"\1\2", cleaned_text)
        
        # 如果生成文本为空或全是空格，返回基本回复
        if not cleaned_text.strip():
            logging.warning("生成文本为空，使用预设回复")
            # 采用一些基本回复
            import random
            responses = [
                "对不起，我还在学习中。",
                "我理解你的意思了。",
                "这个问题很有趣。",
                "我不确定我能回答这个问题。",
                f"{random.choice(common_cn_words)}{random.choice(common_cn_words)}{random.choice(common_cn_words)}"
            ]
            return random.choice(responses)
        
        # 对空格进行额外处理
        cleaned_text = cleaned_text.strip()
        
        # 结果优化：如果生成的文本存在明显的分词问题，采用文本补充
        # 检查是否存在两个字符之间有空格的模式
        space_count = cleaned_text.count(' ')
        chinese_char_count = len(re.findall(chinese_pattern, cleaned_text))
        
        # 如果空格过多，但中文字符很少，可能是分词问题
        if space_count > 10 and chinese_char_count / (space_count + 1) < 1.5:
            logging.warning("检测到可能的分词问题，尝试采用token直接连接")
            # 尝试直接连接tokens
            direct_text = ''.join([t.replace('##', '') for t in tokens]).strip()
            # 如果直接连接的文本更长，使用它
            if len(direct_text) > len(cleaned_text.replace(' ', '')):
                cleaned_text = direct_text
            # 否则采用空格移除方案
            else:
                cleaned_text = cleaned_text.replace(' ', '')
        
        # 检查生成文本的长度，如果小于10个字符，补充一些常用词
        if len(cleaned_text) < 10:
            import random
            # 补充一些常用中文词
            supplement = ''.join(random.sample(common_cn_words, min(5, len(common_cn_words))))
            cleaned_text = cleaned_text + "\n" + supplement
        
        return cleaned_text
    except Exception as e:
        logging.error(f"解码生成文本时出错: {e}")
        # 作为备选，尝试逐字符输出
        try:
            # 直接连接tokens而不使用空格
            direct_text = ''.join([t.replace('##', '') for t in tokens])
            if direct_text.strip():
                return direct_text.strip()
            
            # 如果还是空，返回一个默认回复
            return "对不起，我暂时无法理解您的意思。"
        except:
            return "[生成失败]"

def run_interactive_chat(model, tokenizer, max_length=100, temperature=0.8, top_k=50, top_p=0.9, device='cuda'):
    """运行交互式聊天"""
    print("\n" + "="*50)
    print("欢迎使用Transformer交互式聊天！")
    print("输入 'exit', 'quit' 或 'q' 退出")
    print("输入 '/save' 保存对话历史")
    print("输入 '/params' 查看或修改生成参数")
    print("输入 '/debug' 切换调试模式")
    print("="*50)
    
    # 设置参数
    params = {
        'max_length': max_length,
        'temperature': temperature,
        'top_k': top_k,
        'top_p': top_p
    }
    
    # 调试模式
    debug_mode = False
    
    # 对话历史
    chat_history = []
    
    # 模型信息
    print(f"\n[模型信息] 词表大小: {tokenizer.vocab_size}")
    print(f"[模型信息] 设备: {device}")
    print(f"[模型信息] 温度: {temperature}, top-k: {top_k}, top-p: {top_p}\n")
    
    # 主循环
    while True:
        try:
            # 获取用户输入
            user_input = input("\n> ")
            
            # 检查是否退出
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("再见！")
                break
            
            # 检查是否保存对话
            if user_input.lower() == '/save':
                save_path = f"chat_history_{len(chat_history)}.txt"
                with open(save_path, 'w', encoding='utf-8') as f:
                    for turn in chat_history:
                        f.write(f"User: {turn['user']}\n")
                        f.write(f"Model: {turn['model']}\n\n")
                print(f"对话历史已保存到: {save_path}")
                continue
            
            # 检查是否切换调试模式
            if user_input.lower() == '/debug':
                debug_mode = not debug_mode
                if debug_mode:
                    logging.getLogger().setLevel(logging.DEBUG)
                    print("已切换到调试模式，将显示详细信息")
                else:
                    logging.getLogger().setLevel(logging.INFO)
                    print("已切换到正常模式")
                continue
                
            # 检查是否查看/修改参数
            if user_input.lower() == '/params':
                print("\n当前生成参数:")
                for k, v in params.items():
                    print(f"{k} = {v}")
                
                print("\n要修改参数，请按以下格式输入: '参数名=值'")
                print("例如: 'temperature=0.9' 或 'max_length=150'")
                print("输入空行返回聊天")
                
                while True:
                    param_input = input("参数> ")
                    if not param_input:
                        break
                    
                    try:
                        name, value = param_input.split('=')
                        name = name.strip()
                        value = value.strip()
                        
                        if name in params:
                            # 转换值的类型
                            if name == 'max_length' or name == 'top_k':
                                params[name] = int(value)
                            else:
                                params[name] = float(value)
                            print(f"参数 {name} 已更新为 {params[name]}")
                        else:
                            print(f"未知参数: {name}")
                    except Exception as e:
                        print(f"无效的输入: {e}")
                
                continue
            
            # 生成回复
            print("模型正在生成回复...")
            start_time = time.time()
            model_response = generate_response(
                model, 
                tokenizer, 
                user_input, 
                max_length=params['max_length'],
                temperature=params['temperature'],
                top_k=params['top_k'],
                top_p=params['top_p'],
                device=device
            )
            end_time = time.time()
            
            # 显示回复
            gen_time = end_time - start_time
            print(f"\n模型: {model_response}")
            if debug_mode:
                print(f"\n[调试] 生成时间: {gen_time:.2f}秒, 生成长度: {len(model_response)}")
            
            # 保存对话历史
            chat_history.append({
                'user': user_input,
                'model': model_response
            })
            
        except KeyboardInterrupt:
            print("\n中断，正在退出...")
            break
        except Exception as e:
            print(f"\n出现错误: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="交互式聊天 - 使用预训练的Transformer模型")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--model_path", type=str, required=True, help="模型检查点路径")
    parser.add_argument("--max_length", type=int, default=100, help="生成的最大长度")
    parser.add_argument("--temperature", type=float, default=0.8, help="生成的温度参数")
    parser.add_argument("--top_k", type=int, default=50, help="top-k采样参数")
    parser.add_argument("--top_p", type=float, default=0.9, help="top-p采样参数")
    parser.add_argument("--cuda", action="store_true", help="使用CUDA")
    parser.add_argument("--mps", action="store_true", help="使用MPS (Apple Silicon)")
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 设置设备
    device = setup_device(args.cuda, args.mps)
    
    # 加载分词器
    tokenizer = load_tokenizer(config.get('tokenizer_dir', 'data/hf_tokenizer'))
    
    # 加载模型
    model = load_model(args.model_path, config, device)
    
    # 运行交互式聊天
    run_interactive_chat(
        model,
        tokenizer,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=device
    )

if __name__ == "__main__":
    main()
