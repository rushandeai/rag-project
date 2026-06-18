"""
Step 3: FastAPI 服务 — RESTful 接口
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag import ask_rag

app = FastAPI(title="RAG 知识库问答 API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str


class Source(BaseModel):
    content: str
    rrf_score: float = 0
    rerank_score: float = 0


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source] = []
    retrieval_method: str = "hybrid+rerank"
    status: str = "ok"


@app.get("/")
def root():
    return {
        "message": "RAG 知识库问答系统 API v2.0",
        "pipeline": "BM25 + Dense → RRF 融合 → BGE-Reranker 精排 → DeepSeek 生成",
        "docs": "/docs"
    }


@app.post("/ask", response_model=QueryResponse)
def ask(request: QueryRequest):
    try:
        answer, docs = ask_rag(request.question)
        sources = [
            Source(
                content=doc["content"],
                rrf_score=doc.get("rrf_score", 0),
                rerank_score=doc.get("rerank_score", 0)
            )
            for doc in docs
        ]
        return QueryResponse(answer=answer, sources=sources)
    except Exception as e:
        return QueryResponse(answer=f"出错了: {str(e)}", status="error")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
