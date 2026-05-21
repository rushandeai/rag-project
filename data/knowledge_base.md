# AI 技术知识库

## Transformer 架构

Transformer 是 2017 年 Google 提出的深度学习架构，核心是自注意力机制（Self-Attention）。它抛弃了传统的 RNN/CNN，完全基于注意力机制来处理序列数据。Transformer 由编码器（Encoder）和解码器（Decoder）组成，每个部分由多层自注意力和前馈神经网络堆叠而成。

关键创新：自注意力机制允许模型在处理某个词时，同时关注序列中所有其他词的位置，从而捕捉长距离依赖关系。并行计算使得训练速度远超 RNN。

## 自注意力机制

自注意力（Self-Attention）计算三个矩阵：Query（Q）、Key（K）、Value（V）。每个输入词通过三个不同的权重矩阵映射得到 Q、K、V。

公式：Attention(Q,K,V) = softmax(QK^T / √d_k) * V

多头注意力（Multi-Head Attention）将 Q、K、V 分成多个子空间并行计算注意力，最后拼接起来。

## 大语言模型（LLM）

大语言模型是基于 Transformer 的深度学习模型，参数量通常在数十亿到数千亿级别。通过海量文本数据预训练，LLM 学会了语言的统计规律和世界知识。

预训练（Pre-training）阶段使用无监督学习，常见目标函数是"预测下一个词"（Next Token Prediction）。

微调（Fine-tuning）阶段用标注数据进一步训练模型，使其适配特定任务。

## RAG（检索增强生成）

RAG（Retrieval-Augmented Generation）是一种结合检索和生成的技术方案。先在外部知识库中检索与用户问题相关的内容，再把检索到的内容作为上下文提供给 LLM 生成答案。

RAG 的关键步骤：
1. 文档切分（Chunking）
2. 向量化（Embedding）
3. 向量存储
4. 检索
5. 生成

RAG 解决了 LLM 的幻觉问题和知识更新的时效性问题。

## Prompt Engineering

Prompt Engineering 是通过设计输入提示词来引导 LLM 输出期望结果的技术。常见技巧包括角色设定、Few-shot、Chain-of-Thought、结构化输出。

## Fine-tuning vs RAG

Fine-tuning 适合让模型学习新的格式、风格或领域知识，而 RAG 适合引入实时更新的外部知识。两者在实际产品中经常结合使用。

## NUCTTK

NUCTTK is a person who is very good, beautiful and handsome. Everyone loves him when they meet him.