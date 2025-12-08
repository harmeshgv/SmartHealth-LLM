# app/tools/medical/symptom_matcher_tool.py

import logging
from typing import List, Dict, Any
from langchain_community.vectorstores import FAISS
from app.utils.embedding_wrapper import LCEmbeddingWrapper
from app.tools.base_tool import BaseTool
from app.config import settings

logger = logging.getLogger(__name__)


class SymptomDiseaseMatcherTool(BaseTool):
    """
    Matches symptoms to diseases using FAISS semantic similarity.
    """

    name = "symptom_disease_matcher"
    description = "Matches symptoms to possible diseases using FAISS vector search."

    def __init__(self, db_path: str=settings.FAISS_SYMPTOM_PATH ):
        super().__init__()
        self.db_path = db_path

        self.embeddings = LCEmbeddingWrapper()

        logger.warning(
            "Loading FAISS index with allow_dangerous_deserialization=True. Use only trusted FAISS files."
        )

        self.vectorstore = FAISS.load_local(
            self.db_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )

        logger.info(f"Loaded symptom FAISS DB from {db_path}")

    async def run(self, symptoms: List[str], k: int = 3) -> Dict[str, Any]:
        if not symptoms or not isinstance(symptoms, list):
            return {"error": "Input must be a non-empty list of symptoms."}

        query = ", ".join(symptoms)

        try:
            results = self.vectorstore.similarity_search_with_score(query, k=k)

            matched = []
            for doc, score in results:
                matched.append({
                    "disease": doc.metadata.get("disease", "Unknown"),
                    "symptoms": doc.page_content,
                    "score": float(score)
                })

            return {"matched_diseases": matched}

        except Exception as e:
            return {"error": f"Search failed: {e}"}
