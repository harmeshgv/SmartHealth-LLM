import os
import sys
from typing import List, Tuple

# Ensure project root is on the path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from backend.utils.text_cleaning import Text_Preprocessing
from backend.utils.filtering_with_ner import RemoveUselessWords
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


class DiseaseMatcher:
    def __init__(self, vectorstore_path="Vector/symptom_faiss_db"):
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

    def match(self, user_input: str, top_k: int = 3) -> List[Tuple[str, float, str]]:
        """
        Find top-k matching diseases for the given symptom input.

        Returns:
            List of (disease_name, similarity_score, matching_text)
        """
        processed_input = self._preprocess_text(user_input)
        results = self.vectorstore.similarity_search_with_score(processed_input, k=top_k)

        return [
            (
                doc.metadata.get("disease", "Unknown"),
                score,
                doc.page_content
            )
            for doc, score in results
        ]
