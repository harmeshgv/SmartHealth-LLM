from typing import List, Optional
from dataclasses import dataclass

@dataclass
class EmbeddingObject:
    """Represents an embedding vector."""
    object: str = "embedding"
    embedding: List[float] = None
    index: int = 0

@dataclass
class EmbeddingUsage:
    """Usage statistics for embeddings."""
    prompt_tokens: int = 0
    total_tokens: int = 0

@dataclass
class EmbeddingResponse:
    """Response from embeddings API."""
    object: str = "list"
    data: List[EmbeddingObject] = None
    model: str = ""
    usage: Optional[EmbeddingUsage] = None
    
    def __post_init__(self):
        if self.data is None:
            self.data = []
    
    def model_dump(self):
        """Convert to dictionary for JSON serialization (OpenAI compatibility)."""
        return {
            "object": self.object,
            "data": [
                {
                    "object": emb.object,
                    "embedding": emb.embedding,
                    "index": emb.index
                } for emb in self.data
            ],
            "model": self.model,
            "usage": {
                "prompt_tokens": self.usage.prompt_tokens,
                "total_tokens": self.usage.total_tokens
            } if self.usage else None
        }