"""
Step 1: 文档入库 — 切块 → 向量化 → 存储 → BM25 索引

Chunking 策略说明：
- 按 Markdown 章节（##）做一级切分，保留文档结构
- 对超过 chunk_size 的段落按句号二次切分
- chunk_overlap=50 防止语义在边界被切断
"""
import os
import pickle
import chromadb
import jieba
from rank_bm25 import BM25Okapi
from embedding import SemanticEmbedding

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

DOC_PATH = "data/knowledge_base.md"
CHUNK_SIZE = 300       # 每块最大字符数
CHUNK_OVERLAP = 50     # 相邻块重叠字符数
COLLECTION_NAME = "ai_knowledge"
BM25_PATH = "./chroma_data/bm25_index.pkl"


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    文档切块：先按章节分，再对长段落按句号滑动窗口切分。
    返回带章节标题前缀的 chunk 列表。
    """
    chunks = []

    for section in text.split("\n## "):
        lines = section.strip().split("\n")
        title = lines[0].replace("#", "").strip()
        content = "\n".join(lines[1:]).strip()

        if not content:
            continue

        if len(content) <= chunk_size:
            chunks.append(f"[{title}] {content}")
        else:
            sentences = content.replace("。", "。\n").split("\n")
            current = ""

            for s in sentences:
                if not s.strip():
                    continue
                if len(current) + len(s) < chunk_size:
                    current += s
                else:
                    if current.strip():
                        chunks.append(f"[{title}] {current.strip()}")
                    if len(current) > overlap:
                        current = current[-overlap:] + s
                    else:
                        current = s
            if current.strip():
                chunks.append(f"[{title}] {current.strip()}")

    return chunks


def tokenize(text: str) -> list[str]:
    """中文分词，BM25 需要分词后的 token 列表"""
    return list(jieba.cut(text))


if __name__ == "__main__":
    # 1. 读取文档
    with open(DOC_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"📄 文档长度: {len(text)} 字")

    # 2. 切块
    chunks = chunk_text(text)
    print(f"📦 分成了 {len(chunks)} 个块")

    # 3. 向量化 + 存入 ChromaDB（Dense 检索）
    print("🧠 生成嵌入向量...")
    embed_fn = SemanticEmbedding()
    embeddings = embed_fn(chunks)
    print(f"📐 向量维度: {embed_fn.dim}，模型: BAAI/bge-small-zh-v1.5")

    print("💾 存入 ChromaDB...")
    client = chromadb.PersistentClient(path="./chroma_data")
    # 如果集合已存在就删掉重建（全量重新入库）
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"  已删除旧集合 '{COLLECTION_NAME}'")
    except ValueError:
        pass  # 集合不存在，不需要删
    collection = client.create_collection(name=COLLECTION_NAME)

    ids = [f"chunk_{i}" for i in range(len(chunks))]
    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=[{"index": i} for i in range(len(chunks))]
    )
    print(f"✅ ChromaDB 入库完成！共 {collection.count()} 条记录")

    # 4. 构建 BM25 索引（Sparse 检索）
    print("🔍 构建 BM25 索引...")
    tokenized_chunks = [tokenize(c) for c in chunks]
    bm25 = BM25Okapi(tokenized_chunks)

    os.makedirs(os.path.dirname(BM25_PATH), exist_ok=True)
    with open(BM25_PATH, "wb") as f:
        pickle.dump({
            "chunks": chunks,
            "tokenized": tokenized_chunks
        }, f)
    print(f"✅ BM25 索引已保存到 {BM25_PATH}")
    print(f"\n🎉 入库全部完成！Dense + Sparse 双路索引就绪")
