import os
import sys
from typing import List, Tuple

# Allow importing from parent folders
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from backend.utils.text_cleaning import Text_Preprocessing
from backend.utils.filtering_with_ner import RemoveUselessWords
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


class DiseaseMatcher:
    def __init__(self, vectorstore_path = "Vector/symptom_faiss_db"):
        """
        Initialize the DiseaseMatcher.

        Args:
            vectorstore_path (str): Path to the saved FAISS vectorstore directory.
        """
        self.vectorstore_path = vectorstore_path
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
            allow_dangerous_deserialization=True  # Trust your own FAISS DB
        )

    def _preprocess_text(self, user_input: str) -> str:
        """
        Clean and filter the user symptom input.
        """
        cleaned = self.text_cleaner.go_on(user_input)
        filtered = self.ner_filter.process_entities(cleaned)
        return " ".join(filtered)

    def match(self, user_input: str, top_k: int = 3) -> List[Tuple[str, float, str]]:
        """
        Find top-k matching diseases for given symptom input.

        Returns:
            List of tuples: (disease_name, confidence_score, key_symptom_text)
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
