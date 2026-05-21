"""
自定义嵌入函数 - 使用 sentence-transformers，语义级匹配
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'   # ← 加这行
from sentence_transformers import SentenceTransformer


class SemanticEmbedding:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def __call__(self, texts: list[str]):
        vectors = self.model.encode(texts)
        return [v.tolist() for v in vectors]