# RAG 知识库问答系统

一个手写 RAG pipeline 的练手项目，实现了混合检索 + 重排序。

初衷：学 RAG 的时候不想只调 LangChain 的高级 API，把每一步拆开自己写一遍——Chunking 怎么切？Embedding 选哪个？为什么需要 Reranker？写完才算真懂了。

## 架构

```
用户问题 → BM25 + Dense 双路检索 → RRF 融合(top-20) → BGE-Reranker 精排(top-3) → DeepSeek 生成
```

没有用 LangChain 的 `VectorstoreIndexCreator` 或 `RetrievalQA`，每一层都是手写的。

## 技术选型

| 环节 | 用了什么 | 备注 |
|---|---|---|
| Embedding | BAAI/bge-small-zh-v1.5 | 中文效果好，24MB 本地跑，不用调 API |
| 向量库 | ChromaDB | 轻量，持久化，开发够用。生产应该换 Milvus |
| 关键词检索 | BM25 (rank-bm25 + jieba) | 和 Dense 互补——语义搜不到的关键词靠它 |
| 融合 | Reciprocal Rank Fusion (K=60) | 不用调参数，工业界通用做法 |
| 重排序 | BAAI/bge-reranker-base | Cross-Encoder 精度高，但慢，所以只对 top-20 做 |
| 生成 | DeepSeek Chat | 中文好，便宜，兼容 OpenAI 格式 |
| 后端 | FastAPI | 快，自带 Swagger |
| 前端 | Streamlit | 纯 Python，快速出 UI |

## 评估

自己写了个评估脚本，用 DeepSeek 做 LLM-as-Judge（没用 RAGAS 库，因为 0.4.3 版本和 langchain-community 有兼容性问题）。

8 条测试数据，结果：

| 指标 | 分数 | 说明 |
|---|---|---|
| Faithfulness | 1.00 | 回答全部基于上下文，没编造 |
| Context Precision | 1.00 | 最相关的文档排在前面 |
| Context Recall | 0.96 | 有一条漏了点信息，整体覆盖还行 |

> 注意：知识库只有 9 个 chunk、8 道直球题，这个分数有水分。扩知识库 + 加刁钻问题后会更真实。

## 踩过的坑

- HuggingFace 在国内直连不了，配了 `hf-mirror.com` 镜像。但镜像对 HEAD 请求支持不完整，模型下载时会有 timeout retry（实际能下完，就是慢）。解决方案：第一次下载完成后改 `local_files_only=True`
- DeepSeek 没有 Embeddings API，评估时 Embedding 用的是本地 BGE 模型
- `jieba` 有 pkg_resources 的 deprecation warning，不影响功能，暂时没管

## 快速开始

```bash
# 1. 装依赖
pip install -r requirements.txt

# 2. 配 API Key
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY

# 3. 入库（构建向量库 + BM25 索引）
python ingest.py

# 4. 启动 API（终端 1）
python app.py

# 5. 启动界面（终端 2）
streamlit run ui.py

# 6. 跑评估
python evaluate.py
```

第一次启动会下载 Embedding 模型（~24MB）和 Reranker 模型（~1GB），之后秒启动。

## API

```
POST /ask
{
  "question": "Transformer 为什么比 RNN 好？"
}

→ {
  "answer": "...",
  "sources": [
    {"content": "...", "rrf_score": 0.0328, "rerank_score": 2.8456}
  ]
}
```

## 项目结构

```
├── ingest.py         # 文档切块 + 向量化 + 存储（ChromaDB + BM25）
├── rag.py            # 混合检索 → RRF 融合 → Reranker → 生成
├── embedding.py      # Embedding 模型封装（BGE-small-zh）
├── app.py            # FastAPI 接口
├── ui.py             # Streamlit 聊天界面
├── evaluate.py       # LLM-as-Judge 评估脚本
├── data/
│   ├── knowledge_base.md
│   └── test_questions.json
```

## TODO

- [ ] 多轮对话（目前是单轮问答）
- [ ] 流式输出（SSE）
- [ ] 语义缓存（相似问题直接返回缓存，省 API 费用）
- [ ] Docker 部署
- [ ] 知识库从 9 chunks 扩到实际项目规模
