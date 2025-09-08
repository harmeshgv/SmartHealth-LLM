from langchain_huggingface import HuggingFaceEmbeddings  # updated import
from sentence_transformers import SentenceTransformer

# Singleton pattern to reuse the same embeddings object
class EmbeddingsSingleton:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        return cls._instance


# Simple helper function if you prefer functional style
def get_embeddings():
    return EmbeddingsSingleton.get_instance()
