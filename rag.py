"""
Step 2: RAG 查询 — 混合检索 + 重排序 + 生成

检索流水线：
用户问题
  ├─ BM25 稀疏检索 (关键词匹配) ──┐
  ├─ Dense 稠密检索 (语义匹配)  ──┤
  │                               ▼
  │                          RRF 融合 (top-20)
  │                               │
  │                               ▼
  │                     Cross-Encoder 精排 (top-3)
  │                               │
  │                               ▼
  └──────────────────────→ LLM 生成回答
"""
import os
import pickle
import requests
import chromadb
import jieba
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from embedding import SemanticEmbedding
from dotenv import load_dotenv

load_dotenv()

API_URL = "https://api.deepseek.com/v1/chat/completions"
API_KEY = os.getenv("DEEPSEEK_API_KEY")
COLLECTION_NAME = "ai_knowledge"
BM25_PATH = "./chroma_data/bm25_index.pkl"
TOP_K = 3          # 最终返回给 LLM 的文档数
RETRIEVAL_K = 20    # 粗排候选数
RRF_K = 60          # RRF 平滑参数

# ── 初始化 ──────────────────────────────────────────────
embed_fn = SemanticEmbedding()

client = chromadb.PersistentClient(path="./chroma_data")
collection = client.get_collection(name=COLLECTION_NAME)

# BM25
with open(BM25_PATH, "rb") as f:
    bm25_data = pickle.load(f)
bm25 = BM25Okapi(bm25_data["tokenized"])
bm25_chunks = bm25_data["chunks"]
# 构建 chunk_id → content 的映射表
CHUNK_MAP = {f"chunk_{i}": c for i, c in enumerate(bm25_chunks)}

# Reranker（Cross-Encoder，联合编码 query + doc）
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
try:
    reranker = CrossEncoder("BAAI/bge-reranker-base", max_length=512, local_files_only=True)
except Exception:
    reranker = CrossEncoder("BAAI/bge-reranker-base", max_length=512)


def tokenize(text: str) -> list[str]:
    """中文分词"""
    return list(jieba.cut(text))


# ── 混合检索 ────────────────────────────────────────────

def hybrid_retrieve(query: str, top_k: int = RETRIEVAL_K) -> list[dict]:
    """
    混合检索：BM25（稀疏/关键词） + Dense（稠密/语义） → RRF 融合

    BM25 擅长精确关键词匹配（如「Transformer」「RAG」），
    Dense 擅长语义近似匹配（如「注意力」≈「Attention」）。
    两者互补，RRF 融合后比单独使用任一方式召回率更高。
    """
    # 1. Dense 向量检索
    query_vec = embed_fn([query])
    dense_res = collection.query(query_embeddings=query_vec, n_results=top_k)

    # 2. BM25 稀疏检索
    tokenized_q = tokenize(query)
    bm25_scores = bm25.get_scores(tokenized_q)
    bm25_ranked = sorted(
        [(i, bm25_scores[i]) for i in range(len(bm25_chunks))],
        key=lambda x: x[1], reverse=True
    )[:top_k]

    # 3. RRF 融合
    rrf_scores: dict[str, float] = {}

    for rank, doc_id in enumerate(dense_res["ids"][0]):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (RRF_K + rank + 1)

    for rank, (idx, _) in enumerate(bm25_ranked):
        doc_id = f"chunk_{idx}"
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (RRF_K + rank + 1)

    # 按 RRF 分数排序，取 top_k
    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    results = []
    for doc_id, rrf_score in ranked:
        content = CHUNK_MAP.get(doc_id, "")
        if content:
            results.append({
                "id": doc_id,
                "content": content,
                "rrf_score": round(rrf_score, 4)
            })
    return results


# ── 重排序 ──────────────────────────────────────────────

def rerank(query: str, candidates: list[dict], top_k: int = TOP_K) -> list[dict]:
    """
    Cross-Encoder 重排序。

    为什么需要重排序？
    - 向量检索用双编码器（query 和 doc 分开编码），速度快但精度有限
    - Cross-Encoder 把 query + doc 拼在一起送进 Transformer 联合编码
    - 精度远高于余弦相似度，但速度慢（每对都算一遍）
    - 所以先粗排（hybrid_retrieve 拿 top-20），再精排（rerank 取 top-3）
    """
    if not candidates:
        return []

    pairs = [(query, c["content"]) for c in candidates]
    scores = reranker.predict(pairs, show_progress_bar=False)

    for c, s in zip(candidates, scores):
        c["rerank_score"] = round(float(s), 4)

    ranked = sorted(candidates, key=lambda x: x.get("rerank_score", 0), reverse=True)
    return ranked[:top_k]


# ── 生成 ────────────────────────────────────────────────

def build_prompt(question: str, context_docs: list[dict]) -> tuple[str, str]:
    """构建 LLM 提示词"""
    context = "\n\n---\n\n".join([d["content"] for d in context_docs])
    system = (
        "你是一个 AI 技术知识助手。根据提供的参考资料回答问题。\n"
        "规则：\n"
        "1. 如果资料中有答案，直接引用并标注来源章节\n"
        "2. 如果资料中没有答案，明确说「资料中未找到相关信息」\n"
        "3. 回答简洁，不编造资料中没有的内容"
    )
    user = f"参考资料：\n{context}\n\n---\n\n问题：{question}"
    return system, user


# ── 主入口 ──────────────────────────────────────────────

def ask_rag(question: str) -> tuple[str, list[dict]]:
    """
    RAG 问答：混合检索 → 重排序 → 生成
    返回 (答案, 最终使用的文档列表)
    """
    # Step 1: 混合粗排（拿 20 条候选）
    candidates = hybrid_retrieve(question, top_k=RETRIEVAL_K)

    # Step 2: Cross-Encoder 精排（取 top-3）
    top_docs = rerank(question, candidates, top_k=TOP_K)

    # Step 3: LLM 生成
    system, user = build_prompt(question, top_docs)
    resp = requests.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            "temperature": 0.3
        },
        timeout=30
    )
    resp.raise_for_status()
    answer = resp.json()["choices"][0]["message"]["content"]
    return answer, top_docs


# ── CLI 测试 ────────────────────────────────────────────

if __name__ == "__main__":
    if not API_KEY:
        print("❌ 请设置 DEEPSEEK_API_KEY 环境变量（或创建 .env 文件）")
        exit(1)

    while True:
        q = input("\n❓ 问题 (exit 退出): ")
        if q.lower() == "exit":
            break
        answer, docs = ask_rag(q)
        print(f"\n📝 {answer}")
        print(f"\n📎 检索流水线：粗排 {RETRIEVAL_K} → 精排 {len(docs)} 条")
        for i, d in enumerate(docs):
            rrf = d.get("rrf_score", "N/A")
            rerank_s = d.get("rerank_score", "N/A")
            print(f"  [{i+1}] RRF:{rrf} → Rerank:{rerank_s} | {d['content'][:80]}...")
