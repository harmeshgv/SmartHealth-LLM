import sys
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from collections import defaultdict
from statistics import mean

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from backend.config import VECTOR_DIR

class DiseaseMatcher:
    def __init__(self):
        self.embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = self._load_vectorstore()

    def _load_vectorstore(self) -> FAISS:
        try:
            return FAISS.load_local(
                folder_path=VECTOR_DIR,
                embeddings=self.embedding_model,
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load FAISS index: {e}")

    def match(self, symptoms_input: list, top_k: int = 3):
        disease_scores = defaultdict(list)
        disease_symptoms = defaultdict(set)

        for symptom in symptoms_input:
            results = self.vector_store.similarity_search_with_score(symptom, k=10)

            for doc, score in results:
                disease = doc.metadata.get("disease", "Unknown")
                matched_symptom = doc.page_content.strip()
                disease_scores[disease].append(score)
                disease_symptoms[disease].add(matched_symptom)
        ranked = sorted(
    disease_scores.items(),
    key=lambda x: (
        -len(disease_symptoms[x[0]]),   # 1️⃣ More matched symptoms first
        -mean([1 / (1 + s) for s in x[1]])  # 2️⃣ Higher avg similarity next
    )
)

        top_diseases = [
    (
        disease,
        mean([1 / (1 + s) for s in scores]),  # final avg similarity
        list(disease_symptoms[disease])       # list of matched symptoms
    )
    for disease, scores in ranked[:top_k]
]


        return top_diseases
