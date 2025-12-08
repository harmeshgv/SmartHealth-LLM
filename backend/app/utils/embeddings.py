# app/utils/embeddings.py

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List


class EmbeddingSingleton:
    """
    Loads SentenceTransformer embeddings ONCE.
    Returns vectors for documents & queries.
    """

    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = EmbeddingSingleton()
        return cls._instance

    def __init__(self):
        if EmbeddingSingleton._instance is not None:
            return
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def embed_query(self, text: str):
        if not text:
            return np.zeros((384,), dtype="float32")
        vec = self.model.encode(text)
        return np.array(vec, dtype="float32")

    def embed_documents(self, texts: List[str]):
        if not texts:
            return np.zeros((0, 384), dtype="float32")
        vecs = self.model.encode(texts)
        return np.array(vecs, dtype="float32")
