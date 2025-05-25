# Transformer 模型从零实现

这个项目展示了如何从零开始实现一个 Transformer 模型，包括预训练和微调的完整流程。我们使用PyTorch框架构建了Transformer的各个组件，并结合HuggingFace的分词器实现了一个完整的语言模型训练与推理系统。

## 项目结构

```
transformer-from-scratch/
├── configs/                     # 配置文件
│   ├── pretrain_hf.json          # 标准预训练配置
│   ├── pretrain_hf_small.json    # 小数据集预训练配置
│   └── finetune.json             # 微调配置
├── data/                        # 数据处理
│   ├── conversation_dataset.py    # 对话数据集处理
│   ├── hf_tokenizer.py            # HuggingFace分词器封装
│   └── processed/                 # 预处理后的数据
├── model/                       # 模型实现
│   ├── attention.py               # 多头自注意力实现
│   ├── embeddings.py              # 嵌入层与位置编码
│   ├── feedforward.py             # 前馈神经网络
│   ├── transformer.py             # 完整Transformer模型
│   └── generation.py              # 文本生成相关功能
├── scripts/                     # 训练和推理脚本
│   ├── pretrain_with_hf.py        # 使用HF分词器的预训练脚本
│   ├── finetune_model.py          # 问答数据微调脚本
│   ├── interactive_chat.py        # 交互式聊天界面
│   └── generate_text.py           # 文本生成脚本
├── saved_models/                # 保存的模型检查点
│   ├── pretrain_hf_small/         # 小数据集预训练模型
│   └── finetuned/                 # 微调后的模型
├── requirements.txt             # 项目依赖
└── README.md                    # 项目说明
```

## 项目特点

1. **从零实现Transformer架构**
   - 多头自注意力机制实现
   - 可调整的位置编码
   - 自回归解码器设计
   - 兼容GPU和Apple Silicon加速

2. **HuggingFace集成**
   - 使用HuggingFace的高质量分词器
   - 支持中文分词和处理
   - 兼容bert-base-chinese词表

3. **高性能训练**
   - 混合精度训练(AMP)支持
   - 自动批处理和梯度累积
   - 训练过程中的安全检查
   - 自动处理超出词表范围的token ID

4. **完整的训练和推理流程**
   - 多阶段预训练支持
   - 问答数据的微调功能
   - 交互式聊天界面
   - 参数可调的文本生成

## 训练与微调

### 预训练

项目支持使用对话数据进行预训练，支持以下两种模式：

1. **标准预训练**: 使用大规模对话数据（百万级别样本）
   ```bash
   python scripts/pretrain_with_hf.py --config configs/pretrain_hf.json --cuda --amp
   ```

2. **小数据集训练**: 使用约5000个样本进行快速实验
   ```bash
   python scripts/pretrain_with_hf_small_fixed3.py --config configs/pretrain_hf_small.json --cuda --amp
   ```

### 预训练数据格式

预训练使用对话数据，每行是一个完整的对话文本，格式如下：

```
用户A: 你好，请问一下今天天气如何？ 用户B: 今天天气晴朗，温度适宜。 用户A: 那应该适合出门。 用户B: 是的，非常适合外出活动。
用户A: 你能帮我查一下北京到上海的高铁吗？ 用户B: 当然可以，你要哪天的车次信息？ 用户A: 明天上午的。 用户B: 好的，明天上午有9点，10点30分和11点15分有三起3车次。
```

预训练数据文件的位置在`data/processed/`目录下，包括：
- `dialogue_conversation_train.txt`: 训练集
- `dialogue_conversation_val.txt`: 验证集

这些数据文件包含了完整的对话上下文，而非单个问答对，更适合训练通用的语言模型。

### 微调

支持使用问答对数据进行微调，提升模型在特定对话场景的表现：

```bash
python scripts/finetune_model.py \
  --config configs/finetune.json \
  --model_path saved_models/pretrain_hf_small/checkpoint-epoch-5/pytorch_model.bin \
  --train_file data/raw/train.jsonl \
  --batch_size 8 \
  --epochs 5 \
  --lr 2e-5 \
  --cuda
```

微调数据格式(JSONL)：
```json
{"question": "你好，最近怎么样？", "answer": "你好！我最近还不错，谢谢。"}
{"question": "今天天气如何？", "answer": "今天的天气很晴朗。"}
```

## 文本生成与交互

### 交互式聊天

使用交互式聊天脚本与模型进行实时对话：

```bash
python scripts/interactive_chat.py \
  --config configs/finetune.json \
  --model_path saved_models/finetuned/best_model/pytorch_model.bin \
  --temperature 0.7 \
  --cuda
```

交互式界面支持以下命令：
- `/debug`: 切换调试模式，显示生成详情
- `/params`: 查看和修改生成参数
- `/save`: 保存对话历史

### 单次文本生成

使用生成脚本进行单次文本生成：

```bash
python scripts/generate_text.py \
  --config configs/pretrain_hf_small.json \
  --model_path saved_models/pretrain_hf_small/checkpoint-epoch-5/pytorch_model.bin \
  --prompt "你好，我想" \
  --max_length 50 \
  --temperature 0.8 \
  --cuda
```

## 系统要求与依赖

### 硬件要求
- **GPU训练**: 建议至少8GB显存
- **CPU训练**: 可行但速度较慢
- **Apple Silicon**: 支持M系列芯片加速

### 主要依赖
- PyTorch >= 1.10.0
- Transformers >= 4.20.0
- tqdm
- numpy

## 高级特性

### 混合精度训练

开启混合精度训练可显著提升训练速度，特别是在支持FP16的NVIDIA GPU上：

```bash
python scripts/pretrain_with_hf_small_fixed3.py --cuda --amp
```

### 生成参数调优

在交互式聊天中可以通过`/params`命令实时调整生成参数：
- `temperature`: 控制生成多样性（值越大越随机）
- `top_k`: K值采样参数
- `top_p`: 核采样参数
- `max_length`: 最大生成长度

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - 原始Transformer论文
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/index) - 分词器和模型参考
- [PyTorch Mixed Precision](https://pytorch.org/docs/stable/notes/amp_examples.html) - 混合精度训练指南
