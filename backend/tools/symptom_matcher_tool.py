# backend/tools/disease_matcher_tool.py
import logging
from typing import List, Dict, Any
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

from .base import BaseTool

logger = logging.getLogger(__name__)

class SymptomDiseaseMatcherTool(BaseTool):
    """
    A tool to match a list of symptoms to potential diseases using a vector database.
    """

    name: str = "Symptom to Disease Matcher"
    description: str = (
        "Finds potential diseases that match a given list of symptoms. "
        "Input should be a list of symptom strings."
    )

    def __init__(self, db_path: str, embeddings: Embeddings):
        super().__init__()
        self.db_path = db_path
        self.embeddings = embeddings
        try:
            logger.warning(
                "Loading FAISS index with allow_dangerous_deserialization=True. "
                "This is a security risk if the index file is untrusted."
            )
            self.vectorstore = FAISS.load_local(
                self.db_path, self.embeddings, allow_dangerous_deserialization=True
            )
            logger.info(f"Symptom-disease matcher loaded FAISS index from {db_path}")
        except Exception as e:
            logger.error(f"Failed to load FAISS index from {db_path}: {e}", exc_info=True)
            raise

    async def execute(self, symptoms: List[str], k: int = 3) -> Dict[str, Any]:
        """
        Matches symptoms to diseases.

        Args:
            symptoms: A list of symptoms.
            k: The number of top matches to return.

        Returns:
            A dictionary containing the matched diseases or an error.
        """
        if not symptoms or not isinstance(symptoms, list):
            return {"error": "Input must be a non-empty list of symptoms."}

        query_str = ", ".join(symptoms)

        try:
            results = self.vectorstore.similarity_search_with_score(query_str, k=k)

            matches = []
            for doc, score in results:
                matches.append({
                    "disease": doc.metadata.get("disease", "Unknown"),
                    "symptoms": doc.page_content,
                    "score": score,
                })

            logger.info(f"Found {len(matches)} potential disease matches for symptoms: {symptoms}")
            return {"matched_diseases": matches}
        except Exception as e:
            error_msg = f"An error occurred during similarity search: {e}"
            logger.error(error_msg, exc_info=True)
            return {"error": error_msg}
