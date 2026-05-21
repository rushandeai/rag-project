"""
Step 1: 文档入库
"""
import chromadb
from embedding import SemanticEmbedding


DOC_PATH = "data/test_base.md"
# DOC_PATH = "data/knowledge_base.md"
CHUNK_SIZE = 200
COLLECTION_NAME = "ai_knowledge"

with open(DOC_PATH, "r", encoding="utf-8") as f:
    text = f.read()
print(f"📄 文档长度: {len(text)} 字")

# 分块
chunks = []
for section in text.split("\n## "):
    lines = section.strip().split("\n")
    title = lines[0].replace("#", "").strip()
    content = "\n".join(lines[1:]).strip()
    if len(content) > CHUNK_SIZE:
        sentences = content.replace("。", "。\n").split("\n")
        current = ""
        for s in sentences:
            if len(current) + len(s) < CHUNK_SIZE:
                current += s
            else:
                if current.strip():
                    chunks.append(f"{title}: {current.strip()}")
                current = s
        if current.strip():
            chunks.append(f"{title}: {current.strip()}")
    else:
        if content:
            chunks.append(f"{title}: {content}")
print(f"📦 分成了 {len(chunks)} 个块")

# 纯 Python 嵌入
print("🧠 生成嵌入向量...")
embed_fn = SemanticEmbedding()
embeddings = embed_fn(chunks)
print(f"📐 每个向量维度: {len(embeddings[0])}")

# 存入 ChromaDB
print("💾 存入 ChromaDB...")
client = chromadb.PersistentClient(path="./chroma_data")
try:
    client.delete_collection(COLLECTION_NAME)
except:
    pass
collection = client.create_collection(name=COLLECTION_NAME)
ids = [f"chunk_{i}" for i in range(len(chunks))]
collection.add(ids=ids, documents=chunks, embeddings=embeddings,
               metadatas=[{"index": i} for i in range(len(chunks))])
print(f"✅ 入库完成！共 {collection.count()} 条记录")