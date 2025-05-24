import os
import re
import json
import logging
import multiprocessing
from typing import List, Dict, Optional, Callable, Any, Union
from tqdm import tqdm


def clean_text(text: str) -> str:
    """
    清理文本，包括去除多余空格、特殊字符等
    
    参数:
        text: 输入文本
        
    返回:
        清理后的文本
    """
    # 替换多个空格为单个空格
    text = re.sub(r'\s+', ' ', text)
    
    # 去除HTML标签
    text = re.sub(r'<[^>]+>', '', text)
    
    # 去除URL
    text = re.sub(r'http[s]?://\S+', '', text)
    
    # 删除前后空格
    text = text.strip()
    
    return text


def split_text_into_segments(text: str, max_length: int = 512, overlap: int = 50) -> List[str]:
    """
    将长文本分割成较短的段落
    
    参数:
        text: 输入文本
        max_length: 最大段落长度（按词数计算）
        overlap: 相邻段落的重叠词数
        
    返回:
        段落列表
    """
    words = text.split()
    
    if len(words) <= max_length:
        return [text]
    
    segments = []
    start = 0
    
    while start < len(words):
        end = min(start + max_length, len(words))
        segment = ' '.join(words[start:end])
        segments.append(segment)
        
        # 更新下一段的起始位置（考虑重叠）
        start += max_length - overlap
        
    return segments


def process_raw_file(file_path: str, output_path: str, process_func: Callable[[str], str] = clean_text):
    """
    处理原始文本文件
    
    参数:
        file_path: 输入文件路径
        output_path: 输出文件路径
        process_func: 处理函数，默认为clean_text
    """
    processed_lines = []
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in tqdm(f, desc=f"处理 {os.path.basename(file_path)}"):
            line = line.strip()
            if line:
                processed_line = process_func(line)
                if processed_line:  # 跳过处理后为空的行
                    processed_lines.append(processed_line)
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 写入处理后的文本
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in processed_lines:
            f.write(line + '\n')
    
    logging.info(f"处理完成: {file_path} -> {output_path}, 行数: {len(processed_lines)}")


def process_wiki_dump(input_file: str, output_file: str, min_length: int = 50):
    """
    处理Wikipedia转储文件
    
    参数:
        input_file: 输入文件路径（JSON格式）
        output_file: 输出文件路径
        min_length: 最小文章长度（字符数）
    """
    with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        for line in tqdm(f_in, desc=f"处理 Wikipedia 数据"):
            try:
                article = json.loads(line)
                # 提取文章文本（通常是'text'字段）
                if 'text' in article:
                    text = article['text']
                    # 清理和分段
                    text = clean_text(text)
                    if len(text) >= min_length:
                        segments = split_text_into_segments(text)
                        # 写入每个段落
                        for segment in segments:
                            if len(segment) >= min_length:
                                f_out.write(segment + '\n')
            except json.JSONDecodeError:
                continue
    
    logging.info(f"Wikipedia 数据处理完成: {input_file} -> {output_file}")


def process_directory(input_dir: str, output_dir: str, file_ext: str = '.txt', 
                     process_func: Callable[[str], str] = clean_text,
                     num_workers: int = None):
    """
    并行处理目录中的所有文件
    
    参数:
        input_dir: 输入目录
        output_dir: 输出目录
        file_ext: 文件扩展名
        process_func: 处理函数
        num_workers: 并行处理的工作进程数，默认为CPU核心数
    """
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 收集所有匹配的文件
    files_to_process = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(file_ext):
                input_path = os.path.join(root, file)
                # 保持相对路径结构
                rel_path = os.path.relpath(input_path, input_dir)
                output_path = os.path.join(output_dir, rel_path)
                
                # 确保输出文件的目录存在
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                files_to_process.append((input_path, output_path))
    
    logging.info(f"发现 {len(files_to_process)} 个文件需要处理")
    
    # 创建进程池并行处理
    with multiprocessing.Pool(num_workers) as pool:
        args = [(input_path, output_path, process_func) 
                for input_path, output_path in files_to_process]
        
        # 使用starmap处理多个参数
        list(tqdm(pool.starmap(process_raw_file, args), 
                 total=len(args), 
                 desc="处理文件"))
    
    logging.info(f"目录处理完成: {input_dir} -> {output_dir}")


def create_train_val_split(input_file: str, train_file: str, val_file: str, val_ratio: float = 0.1, shuffle: bool = True):
    """
    将数据集分割为训练集和验证集
    
    参数:
        input_file: 输入文件路径
        train_file: 训练集输出路径
        val_file: 验证集输出路径
        val_ratio: 验证集比例
        shuffle: 是否打乱数据
    """
    # 读取所有行
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    # 打乱数据（可选）
    if shuffle:
        import random
        random.shuffle(lines)
    
    # 计算分割点
    split_idx = int(len(lines) * (1 - val_ratio))
    train_lines = lines[:split_idx]
    val_lines = lines[split_idx:]
    
    # 写入训练集
    with open(train_file, 'w', encoding='utf-8') as f:
        for line in train_lines:
            f.write(line + '\n')
    
    # 写入验证集
    with open(val_file, 'w', encoding='utf-8') as f:
        for line in val_lines:
            f.write(line + '\n')
    
    logging.info(f"数据集分割完成: {input_file} -> {train_file} ({len(train_lines)}行), {val_file} ({len(val_lines)}行)")


def prepare_chinese_corpus(input_file: str, output_file: str):
    """
    中文语料预处理
    
    参数:
        input_file: 输入文件路径
        output_file: 输出文件路径
    """
    def process_chinese_text(text):
        # 去除非中文字符（保留中文、字母、数字和基本标点）
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9.,!?;:，。！？；：、]', ' ', text)
        # 替换多个空格为单个空格
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    process_raw_file(input_file, output_file, process_func=process_chinese_text)


def prepare_classification_data(input_file: str, output_file: str, text_field: str, label_field: str):
    """
    准备文本分类数据
    
    参数:
        input_file: 输入文件路径（CSV或TSV格式）
        output_file: 输出文件路径（JSONL格式）
        text_field: 文本字段名
        label_field: 标签字段名
    """
    import csv
    
    # 确定分隔符
    if input_file.endswith('.csv'):
        delimiter = ','
    elif input_file.endswith('.tsv'):
        delimiter = '\t'
    else:
        raise ValueError(f"不支持的文件格式: {input_file}，请使用CSV或TSV格式")
    
    # 读取CSV/TSV文件
    with open(input_file, 'r', encoding='utf-8', newline='') as f_in:
        reader = csv.DictReader(f_in, delimiter=delimiter)
        
        # 确保必要字段存在
        fieldnames = reader.fieldnames
        if text_field not in fieldnames or label_field not in fieldnames:
            raise ValueError(f"必要字段不存在，请确保数据包含 {text_field} 和 {label_field} 字段")
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # 转换为JSONL格式
        with open(output_file, 'w', encoding='utf-8') as f_out:
            for row in tqdm(reader, desc=f"处理 {os.path.basename(input_file)}"):
                text = row[text_field].strip()
                label = row[label_field].strip()
                
                if text and label:  # 跳过空文本或空标签
                    # 清理文本
                    text = clean_text(text)
                    if text:  # 确保清理后文本不为空
                        json_obj = {
                            'text': text,
                            'label': label
                        }
                        f_out.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
    
    logging.info(f"分类数据准备完成: {input_file} -> {output_file}")


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 使用示例
    logging.info("数据预处理模块。请通过导入此模块并调用相关函数来处理数据。")
