"""
Step 3: FastAPI 服务
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag import ask_rag

app = FastAPI(title="RAG 知识库问答 API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    status: str = "ok"

@app.get("/")
def root():
    return {"message": "RAG 知识库问答系统 API"}

@app.post("/ask", response_model=QueryResponse)
def ask(request: QueryRequest):
    try:
        answer = ask_rag(request.question)
        return QueryResponse(answer=answer)
    except Exception as e:
        return QueryResponse(answer=f"出错了: {str(e)}", status="error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)