import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'   # ← 加这行
from sentence_transformers import SentenceTransformer

import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

vec_rag = model.encode(["RAG is retrieval augmented generation"])[0]
vec_cat = model.encode(["I love my cat"])[0]

# Cosine similarity (1 = identical, 0 = unrelated)
def similar(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print(similar(vec_rag, model.encode(["dog"])[0]))  # ~0.9
print(similar(vec_cat, model.encode(["I love"])[0]))      # ~0.3