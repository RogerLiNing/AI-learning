import os
import json
import torch
import logging
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import sys
from sklearn.metrics import classification_report, accuracy_score, f1_score

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.transformer import TransformerLM
from data.tokenizer import Tokenizer, BPETokenizer
from data.dataset import ClassificationDataset, create_dataloader
from scripts.finetune import TransformerForClassification


def setup_logging(log_dir):
    """设置日志"""
    os.makedirs(log_dir, exist_ok=True)
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "evaluate.log")),
            logging.StreamHandler()
        ]
    )


def load_model(model_path, model_type, num_classes, vocab_size, device='cuda'):
    """
    加载模型
    
    参数:
        model_path: 模型路径
        model_type: 模型类型，'lm'或'classification'
        num_classes: 分类类别数（仅用于分类模型）
        vocab_size: 词表大小
        device: 设备
        
    返回:
        加载的模型
    """
    # 加载检查点
    checkpoint = torch.load(model_path, map_location=device)
    
    # 创建模型
    if model_type == 'lm':
        model = TransformerLM(
            vocab_size=vocab_size,
            d_model=512,  # 应与训练配置一致
            n_layers=6,   # 应与训练配置一致
            n_heads=8,    # 应与训练配置一致
            d_ff=2048     # 应与训练配置一致
        )
    elif model_type == 'classification':
        # 先创建基础模型
        base_model = TransformerLM(
            vocab_size=vocab_size,
            d_model=512,  # 应与训练配置一致
            n_layers=6,   # 应与训练配置一致
            n_heads=8,    # 应与训练配置一致
            d_ff=2048     # 应与训练配置一致
        )
        
        # 创建分类模型
        model = TransformerForClassification(
            pretrained_model=base_model,
            num_classes=num_classes,
            dropout=0.1    # 应与训练配置一致
        )
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    # 加载模型参数
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 将模型移至设备
    model.to(device)
    
    # 设置为评估模式
    model.eval()
    
    logging.info(f"从 {model_path} 加载模型")
    return model


def evaluate_classification(model, data_loader, device):
    """
    评估分类模型
    
    参数:
        model: 模型实例
        data_loader: 数据加载器
        device: 设备
        
    返回:
        评估结果字典
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="评估"):
            # 将数据移至设备
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # 前向传播
            logits = model(input_ids, attention_mask)
            
            # 获取预测结果
            _, preds = torch.max(logits, 1)
            
            # 收集预测和标签
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算评估指标
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    report = classification_report(all_labels, all_preds, digits=4)
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'classification_report': report
    }


def evaluate_language_model(model, data_loader, device):
    """
    评估语言模型
    
    参数:
        model: 模型实例
        data_loader: 数据加载器
        device: 设备
        
    返回:
        评估结果字典
    """
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="评估"):
            # 将数据移至设备
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 前向传播
            outputs = model(input_ids, attention_mask)
            
            # 计算损失
            logits = outputs.view(-1, outputs.size(-1))
            labels = labels.view(-1)
            
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='sum')
            loss = loss_fct(logits, labels)
            
            # 累计损失和token数量
            total_loss += loss.item()
            total_tokens += (labels != -100).sum().item()
    
    # 计算困惑度（perplexity）
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    
    return {
        'loss': avg_loss,
        'perplexity': perplexity
    }


def generate_text(model, tokenizer, prompt, max_length=50, temperature=1.0, top_k=0, top_p=0.9, device='cuda'):
    """
    使用语言模型生成文本
    
    参数:
        model: 模型实例
        tokenizer: 分词器实例
        prompt: 提示文本
        max_length: 生成的最大长度
        temperature: 采样温度
        top_k: top-k采样参数
        top_p: top-p采样参数
        device: 设备
        
    返回:
        生成的文本
    """
    # 编码提示
    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    input_ids = torch.tensor([input_ids]).to(device)
    
    # 生成文本
    generated = model.generate(
        input_ids,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p
    )[0]
    
    # 解码生成的文本
    generated_text = tokenizer.decode(generated.tolist(), skip_special_tokens=True)
    
    return generated_text


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Transformer 评估脚本")
    parser.add_argument("--model", type=str, required=True, help="模型路径")
    parser.add_argument("--tokenizer", type=str, required=True, help="分词器路径")
    parser.add_argument("--test_data", type=str, help="测试数据路径")
    parser.add_argument("--model_type", type=str, default="classification", choices=["lm", "classification"], help="模型类型")
    parser.add_argument("--num_classes", type=int, default=2, help="分类类别数")
    parser.add_argument("--batch_size", type=int, default=16, help="批次大小")
    parser.add_argument("--max_seq_len", type=int, default=512, help="最大序列长度")
    parser.add_argument("--generate", action="store_true", help="生成文本（仅适用于语言模型）")
    parser.add_argument("--prompt", type=str, help="文本生成的提示")
    parser.add_argument("--output_dir", type=str, default="./results", help="输出目录")
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.output_dir)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"使用设备: {device}")
    
    # 加载分词器
    if not os.path.exists(args.tokenizer):
        raise ValueError(f"分词器路径不存在: {args.tokenizer}")
    
    # 根据文件扩展名判断分词器类型
    if 'bpe' in args.tokenizer:
        tokenizer = BPETokenizer.from_file(args.tokenizer)
    else:
        tokenizer = Tokenizer.from_file(args.tokenizer)
    
    logging.info(f"从 {args.tokenizer} 加载分词器")
    
    # 加载模型
    model = load_model(
        args.model, 
        args.model_type, 
        args.num_classes, 
        len(tokenizer.token2id), 
        device
    )
    
    # 文本生成（仅适用于语言模型）
    if args.generate and args.model_type == 'lm':
        if not args.prompt:
            raise ValueError("生成文本时必须提供提示 (--prompt)")
        
        logging.info(f"使用提示 '{args.prompt}' 生成文本...")
        generated_text = generate_text(
            model, 
            tokenizer, 
            args.prompt, 
            max_length=100, 
            temperature=0.7, 
            top_p=0.9, 
            device=device
        )
        
        logging.info(f"生成的文本: {generated_text}")
        
        # 保存生成的文本
        output_file = os.path.join(args.output_dir, "generated_text.txt")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"提示: {args.prompt}\n\n")
            f.write(f"生成文本: {generated_text}")
        
        logging.info(f"生成的文本已保存至: {output_file}")
        return
    
    # 模型评估
    if not args.test_data or not os.path.exists(args.test_data):
        raise ValueError(f"测试数据路径不存在: {args.test_data}")
    
    # 加载测试数据
    if args.model_type == 'classification':
        # 分类数据集
        test_dataset = ClassificationDataset.from_file(
            args.test_data,
            tokenizer=tokenizer,
            max_length=args.max_seq_len
        )
    else:
        # 语言模型数据集
        from data.dataset import MLMDataset
        test_dataset = MLMDataset(
            texts=[],  # 将在from_file中加载
            tokenizer=tokenizer,
            max_length=args.max_seq_len
        )
        test_dataset = test_dataset.__class__.from_file(
            args.test_data,
            tokenizer=tokenizer,
            max_length=args.max_seq_len
        )
    
    # 创建数据加载器
    test_loader = create_dataloader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # 评估模型
    if args.model_type == 'classification':
        results = evaluate_classification(model, test_loader, device)
        
        logging.info(f"分类评估结果:")
        logging.info(f"准确率: {results['accuracy']:.4f}")
        logging.info(f"F1分数: {results['f1_score']:.4f}")
        logging.info(f"分类报告:\n{results['classification_report']}")
        
        # 保存结果
        output_file = os.path.join(args.output_dir, "classification_results.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'accuracy': float(results['accuracy']),
                'f1_score': float(results['f1_score']),
                'classification_report': results['classification_report']
            }, f, indent=2)
    else:
        results = evaluate_language_model(model, test_loader, device)
        
        logging.info(f"语言模型评估结果:")
        logging.info(f"损失: {results['loss']:.4f}")
        logging.info(f"困惑度: {results['perplexity']:.4f}")
        
        # 保存结果
        output_file = os.path.join(args.output_dir, "lm_results.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'loss': float(results['loss']),
                'perplexity': float(results['perplexity'])
            }, f, indent=2)
    
    logging.info(f"评估结果已保存至: {output_file}")


if __name__ == "__main__":
    main()
