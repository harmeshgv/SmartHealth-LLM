import os
import sys
from typing import List, Tuple
from collections import defaultdict
from statistics import mean
from langchain_core.documents import Document

# Ensure project root is on the path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from backend.utils.text_cleaning import Text_Preprocessing
from backend.utils.filtering_with_ner import RemoveUselessWords
from backend.utils.ner import NER
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from backend.config import VECTOR_DIR

class DiseaseMatcher:
    def __init__(self, vectorstore_path=VECTOR_DIR):
        """
        Initialize the DiseaseMatcher and load the FAISS vectorstore.
        """
        # Convert relative path to absolute path
        self.vectorstore_path = os.path.abspath(vectorstore_path)

        # Validate vectorstore existence
        index_file = os.path.join(self.vectorstore_path, "index.faiss")
        if not os.path.exists(index_file):
            raise FileNotFoundError(f"❌ FAISS index not found at: {index_file}")

        self.text_cleaner = Text_Preprocessing()
        self.ner_filter = RemoveUselessWords()
        self.embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = self._load_vectorstore()
        self.ner = NER()

    def _load_vectorstore(self) -> FAISS:
        """
        Load the FAISS vectorstore from the given path.
        """
        return FAISS.load_local(
            folder_path=self.vectorstore_path,
            embeddings=self.embedding_model,
            allow_dangerous_deserialization=True  # Only if you trust the DB
        )

    def _preprocess_text(self, user_input: str) -> str:
        """
        Clean and filter the user symptom input using NER and custom text cleaning.
        """
        cleaned = self.text_cleaner.go_on(user_input)
        filtered = self.ner_filter.process_entities(cleaned)
        return " ".join(filtered)
        #filtered = self.ner.extract_entities(user_input)
        #return " ".join(filtered)

    def match(self, user_input: str, top_k: int = 3, similarity_threshold: float = 0.7):
        processed_input = self._preprocess_text(user_input)
        input_symptoms = processed_input.split()  # each word embedded separately

        disease_scores = defaultdict(list)
        disease_symptoms = defaultdict(set)

        for symptom in input_symptoms:
            results = self.vectorstore.similarity_search_with_score(symptom, k=10)

            for doc, score in results:
                if score < similarity_threshold:
                    continue  # Ignore weak matches
                
                disease = doc.metadata.get("disease", "Unknown")
                matched_symptom = doc.page_content.strip()
                disease_scores[disease].append(score)
                disease_symptoms[disease].add(matched_symptom)

        ranked = sorted(
            disease_scores.items(),
            key=lambda x: (-len(disease_symptoms[x[0]]), -mean(x[1]))
        )

        top_diseases = [
            (disease, mean(scores), list(disease_symptoms[disease]))
            for disease, scores in ranked[:top_k]
        ]

        return top_diseases


