# disease_match.py

from llama_index.core import Document, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import pandas as pd
import os
import sys

# --------- 1. Load CSV path ---------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from backend.config import MAYO_CSV


class DiseaseMatcherAgent:
    def __init__(self, top_k=3):
        self.top_k = top_k
        self.docs = []
        self.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Load and index data
        self._load()
        self.index = VectorStoreIndex.from_documents(self.docs, embed_model=self.embed_model)
        self.retriever = self.index.as_retriever(similarity_top_k=self.top_k)

    def _load(self):
        df = pd.read_csv(MAYO_CSV)
        for _, row in df.iterrows():
            disease_name = row['disease']
            symptoms_text = row['Symptoms']
            text = f"Disease: {disease_name}\nSymptoms: {symptoms_text}"
            self.docs.append(Document(text=text, metadata={"disease": disease_name}))

    def match(self, query, top_k=None):
        if top_k:
            self.retriever.similarity_top_k = top_k
        results = self.retriever.retrieve(query)
        return [r.metadata["disease"] for r in results]



