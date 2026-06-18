# AI 技术知识库

## Transformer 架构

Transformer 是 2017 年 Google 提出的深度学习架构，核心是自注意力机制（Self-Attention）。它抛弃了传统的 RNN/CNN，完全基于注意力机制来处理序列数据。Transformer 由编码器（Encoder）和解码器（Decoder）组成，每个部分由多层自注意力和前馈神经网络堆叠而成。

关键创新：自注意力机制允许模型在处理某个词时，同时关注序列中所有其他词的位置，从而捕捉长距离依赖关系。并行计算使得训练速度远超 RNN。

## 自注意力机制

自注意力（Self-Attention）计算三个矩阵：Query（Q）、Key（K）、Value（V）。每个输入词通过三个不同的权重矩阵映射得到 Q、K、V。

公式：Attention(Q,K,V) = softmax(QK^T / √d_k) * V

多头注意力（Multi-Head Attention）将 Q、K、V 分成多个子空间并行计算注意力，最后拼接起来。每个头可以关注不同位置的不同表示子空间。

## 大语言模型（LLM）

大语言模型是基于 Transformer 的深度学习模型，参数量通常在数十亿到数千亿级别。通过海量文本数据预训练，LLM 学会了语言的统计规律和世界知识。

预训练（Pre-training）阶段使用无监督学习，常见目标函数是"预测下一个词"（Next Token Prediction）。微调（Fine-tuning）阶段用标注数据进一步训练模型，使其适配特定任务。

主流模型包括 GPT 系列（OpenAI）、Claude 系列（Anthropic）、Llama 系列（Meta）、Qwen 系列（阿里）、DeepSeek 系列（深度求索）。

## RAG（检索增强生成）

RAG（Retrieval-Augmented Generation）是一种结合检索和生成的技术方案。先在外部知识库中检索与用户问题相关的内容，再把检索到的内容作为上下文提供给 LLM 生成答案。

RAG 的关键步骤：
1. 文档切分（Chunking）：将长文档切分为适合检索的小块
2. 向量化（Embedding）：将文本块转换为语义向量
3. 向量存储：存入向量数据库（如 ChromaDB、Milvus）
4. 检索（Retrieval）：对用户查询做向量检索
5. 重排序（Reranking）：对粗排结果做精细排序
6. 生成（Generation）：LLM 基于上下文生成答案

RAG 解决了 LLM 的两个核心问题：幻觉（Hallucination）和知识时效性。

## Embedding 模型

Embedding 模型将文本映射为固定维度的向量，语义相似的文本在向量空间中距离更近。主流中文 Embedding 模型包括 BGE 系列（BAAI）、M3E 系列、text2vec 系列。

BGE（BAAI General Embedding）是智源研究院推出的开源 Embedding 模型，在 C-MTEB 中文基准测试中排名前列。bge-small-zh-v1.5 为轻量版本（24MB），适合本地部署。

## 向量数据库

向量数据库专门存储和检索高维向量，核心功能是近似最近邻搜索（ANN）。常用向量数据库包括 ChromaDB（轻量级，适合原型开发）、Milvus（分布式，适合生产环境）、FAISS（Meta 开源的向量检索库）、Qdrant（高性能 Rust 实现）。

## Prompt Engineering

Prompt Engineering 是通过设计输入提示词来引导 LLM 输出期望结果的技术。常见技巧包括角色设定、Few-shot、Chain-of-Thought（CoT）、结构化输出。

## Fine-tuning vs RAG

Fine-tuning 适合让模型学习新的格式、风格或领域知识。RAG 适合引入实时更新的外部知识。两者在实际产品中经常结合使用：用 Fine-tuning 让模型学会特定领域的表达方式，用 RAG 引入最新的事实数据。
