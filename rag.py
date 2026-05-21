"""
Step 2: RAG 查询
"""
import os
import requests
import chromadb
from embedding import SemanticEmbedding


API_URL = "https://api.deepseek.com/v1/chat/completions"
API_KEY = os.getenv('DEEPSEEK_API_KEY', 'sk-282cca9ee9f0421384ac37f1194d8c7f')
COLLECTION_NAME = "ai_knowledge"

embed_fn = SemanticEmbedding()
client = chromadb.PersistentClient(path="./chroma_data")
collection = client.get_collection(name=COLLECTION_NAME)


def ask_rag(question: str) -> str:
    query_vec = embed_fn([question])
    results = collection.query(query_embeddings=query_vec, n_results=3)
    context = "\n\n".join(results["documents"][0])

    resp = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json={
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "你是一个AI技术问答助手。根据资料回答问题。"},
                {"role": "user", "content": f"参考资料：\n{context}\n\n---\n\n问题：{question}"}
            ],
            "temperature": 0.3
        }
    )
    return resp.json()["choices"][0]["message"]["content"]


if __name__ == "__main__":
    while True:
        q = input("\n❓ 问题 (exit退出): ")
        if q.lower() == "exit":
            break
        print("📝", ask_rag(q))