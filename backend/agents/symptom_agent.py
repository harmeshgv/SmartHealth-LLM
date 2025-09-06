import pandas as pd
import os
import sys
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from backend.config import FAISS_INDEX

# Load embedding model

class DiseaseMatcher:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        self.vectorstore = FAISS.load_local(FAISS_INDEX, self.embeddings, allow_dangerous_deserialization=True)

    def match(self, query):
        query = ",".join(query)
        results = self.vectorstore.similarity_search(query, k=3)
        final = "symptom matched disease: "
        for r in results:
            print(f"Disease: {r.metadata['disease']}, Text: {r.page_content}")
            final+= f"{r.metadata['disease']},"
        return final








