# Transformer 架构可视化

这个项目提供了 Transformer 架构的交互式可视化，帮助理解这一强大的神经网络架构。Transformer 是谷歌在 2017 年提出的，基于论文 "Attention is All You Need"，已成为现代大语言模型的基础架构。

## 功能特点

- **整体架构可视化**：展示 Transformer 的编码器 (Encoder) 和解码器 (Decoder) 组成结构
- **编码器/解码器详解**：深入了解编码器和解码器模块的内部构造
- **自注意力机制演示**：展示 Self-Attention 的计算过程和重要性
- **多头注意力机制**：解释 Multi-Head Attention 如何并行捕获不同的特征
- **翻译演示**：展示 Transformer 如何处理"动物没有过马路，因为它太累了"这样的句子，特别关注代词"它"如何通过注意力机制正确关联到"动物"

## 如何使用

1. 在浏览器中打开 `index.html` 文件
2. 使用页面顶部的按钮在不同视图之间切换：
   - 整体架构
   - 编码器 (Encoder)
   - 解码器 (Decoder)
   - 自注意力机制 (Self-Attention)
   - 多头注意力 (Multi-Head)
   - 翻译演示
3. 探索交互元素：
   - 点击注意力权重矩阵中的单元格可以看到对应的注意力连接
   - 在翻译演示中使用按钮播放动画或高亮特定关系

## 技术细节

本项目使用纯 HTML、CSS 和 JavaScript 实现，无需任何外部库或框架。可视化元素使用 CSS Grid、Flexbox 和 CSS 动画实现。

## Transformer 关键概念

- **自注意力机制 (Self-Attention)**：允许模型关注输入序列中的不同部分，直接建立远距离依赖关系
- **多头注意力 (Multi-Head Attention)**：并行运行多个注意力机制，捕获不同的表示子空间
- **残差连接 (Residual Connection)**：防止网络退化，帮助训练更深的网络
- **层归一化 (Layer Normalization)**：稳定训练过程

## 参考资料

- [Attention is All You Need](https://arxiv.org/abs/1706.03762) - 原始 Transformer 论文
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - Transformer 可视化解释
