# app/utils/embedding_wrapper.py

from langchain_core.embeddings import Embeddings
from app.utils.embeddings import EmbeddingSingleton


class LCEmbeddingWrapper(Embeddings):
    """
    LangChain-compatible embedding wrapper.
    Allows FAISS to use our custom embedding model.
    """

    def __init__(self):
        self.model = EmbeddingSingleton.get_instance()

    def embed_query(self, text: str):
        return self.model.embed_query(text)

    def embed_documents(self, texts: list[str]):
        return self.model.embed_documents(texts)
