# RAG 知识库问答系统

基于 RAG（Retrieval-Augmented Generation）架构的智能问答系统。  
支持文档自动切块 → 语义检索 → LLM 生成回答，全流程手写实现。

## 技术栈

- **Embedding：** Sentence-Transformer（all-MiniLM-L6-v2）
- **向量数据库：** ChromaDB
- **LLM：** DeepSeek API（兼容 OpenAI 格式）
- **后端：** FastAPI
- **前端：** Streamlit

## 快速开始

```bash

pip install -r requirements.txt

python ingest.py    # 导入知识库

python app.py       # 启动 API

浏览器打开 http://localhost:8000/docs 测试。

项目结构
BASH


├── ingest.py       ← 文档入库（切块 → 向量化 → 存储）

├── rag.py          ← RAG 查询（检索 → 生成）

├── app.py          ← FastAPI REST 接口

├── ui.py           ← Streamlit 聊天界面

├── embedding.py    ← 自定义嵌入函数

└── data/

   └── knowledge_base.md  ← 知识库文档
License 

MIT
