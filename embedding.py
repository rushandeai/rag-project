"""
Embedding 模块：文本 → 语义向量

选型说明：
- 使用 BAAI/bge-small-zh-v1.5，专为中文优化的 Embedding 模型
- BGE 系列在 C-MTEB 中文基准测试中排名前列
- small 版本（24MB）速度与质量均衡，适合本地开发
- 国内网络优先从 HuggingFace 镜像下载，缓存后跳过在线检查
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from sentence_transformers import SentenceTransformer


class SemanticEmbedding:
    """将文本列表转换为语义向量"""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5"):
        # 先尝试从本地缓存加载（跳过在线检查，国内网络不稳定）
        # 如果本地没有缓存，回退到在线下载
        try:
            self.model = SentenceTransformer(model_name, local_files_only=True)
        except Exception:
            self.model = SentenceTransformer(model_name)

        # get_sentence_embedding_dimension → get_embedding_dimension（新版更名）
        try:
            self._dim = self.model.get_embedding_dimension()
        except AttributeError:
            self._dim = self.model.get_sentence_embedding_dimension()

    @property
    def dim(self) -> int:
        return self._dim

    def __call__(self, texts: list[str]) -> list[list[float]]:
        if isinstance(texts, str):
            texts = [texts]
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True
        )
        return [v.tolist() for v in vectors]
