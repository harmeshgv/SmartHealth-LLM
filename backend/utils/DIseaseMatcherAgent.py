import spacy
import pickle
import os
import sys
from langchain_community.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.config import MAYO_CSV, VECTOR_DIR

class DiseaseMatcherAgent:
    def __init__(self, db_path=VECTOR_DIR + "disease_db.pkl", spacy_model="en_ner_bc5cdr_md", embedder_model="all-MiniLM-L6-v2", threshold=0.7):
        # Load SpaCy
        self.nlp = spacy.load(spacy_model)

        # Load embedder
        self.embedder = HuggingFaceEmbeddings(model_name=embedder_model)

        # Load prebuilt DB
        with open(db_path, "rb") as f:
            self.disease_db = pickle.load(f)

        self.threshold = threshold
        print(f"✅ Loaded disease DB with {len(self.disease_db)} entries.")

    def extract_symptoms(self, text):
        doc = self.nlp(text)
        symptoms = list({ent.text.lower().strip() for ent in doc.ents})
        return symptoms

    def match(self, user_symptoms_text, top_k=3):
        user_symptoms = self.extract_symptoms(user_symptoms_text)
        if not user_symptoms:
            return []

        user_embeddings = self.embedder.embed_documents(user_symptoms)
        results = []

        for disease, data in self.disease_db.items():
            scores = cosine_similarity(user_embeddings, data["embeddings"])
            max_sims = scores.max(axis=1)
            coverage = (max_sims > self.threshold).mean()
            avg_similarity = max_sims.mean()
            results.append((disease, coverage, avg_similarity))

        results.sort(key=lambda x: (-x[1], -x[2]))
        return results[:top_k]
