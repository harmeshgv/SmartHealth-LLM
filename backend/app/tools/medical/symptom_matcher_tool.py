# app/tools/medical/symptom_matcher_tool.py
import time
import logging
import csv
import os
from typing import List, Dict, Any
from langchain_community.vectorstores import FAISS
from app.utils.embedding_wrapper import LCEmbeddingWrapper
from app.tools.base_tool import BaseTool
from app.config import settings
from app.core.metrics_tracker import metrics_tracker

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

        self.vectorstore = self._load_or_rebuild_vectorstore()

        logger.info(f"Loaded symptom FAISS DB from {db_path}")

    def _load_or_rebuild_vectorstore(self) -> FAISS:
        try:
            return FAISS.load_local(
                self.db_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        except Exception as exc:
            # Legacy FAISS pickles created with older pydantic/langchain can fail
            # with "__fields_set__" errors after dependency upgrades.
            logger.warning(
                "Failed to load symptom FAISS DB from %s (%s). Rebuilding from CSV.",
                self.db_path,
                str(exc),
            )
            return self._rebuild_vectorstore()

    def _rebuild_vectorstore(self) -> FAISS:
        csv_path = settings.DISEASE_INFO_PATH
        disease_symptoms: Dict[str, List[str]] = {}

        with open(csv_path, mode="r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                disease = (row.get("disease") or "").strip()
                symptoms_str = (row.get("updated") or "").strip()
                if not disease or not symptoms_str:
                    continue
                symptoms = [s.strip() for s in symptoms_str.split(",") if s.strip()]
                if symptoms:
                    disease_symptoms[disease] = symptoms

        texts = ["; ".join(symptoms) for symptoms in disease_symptoms.values()]
        metadatas = [{"disease": name} for name in disease_symptoms.keys()]
        if not texts:
            raise RuntimeError(f"No symptom text found in {csv_path}; cannot rebuild FAISS DB.")

        vectorstore = FAISS.from_texts(texts, self.embeddings, metadatas=metadatas)
        os.makedirs(self.db_path, exist_ok=True)
        vectorstore.save_local(self.db_path)
        logger.info("Rebuilt and saved symptom FAISS DB at %s", self.db_path)
        return vectorstore

    async def run(self, symptoms: List[str], k: int = 3, run_id: str = None) -> Dict[str, Any]:
        start_time = time.time()
        logger.info("Tool run started", extra={"run_id": run_id, "tool": self.name})

        if not symptoms or not isinstance(symptoms, list):
            logger.warning("Input must be a non-empty list of symptoms", extra={"run_id": run_id, "tool": self.name})
            metrics_tracker.record_tool_event(
                run_id=run_id, tool_name=self.name, source="vector_db", success=False, error=True
            )
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
            
            output = {"matched_diseases": matched}

            end_time = time.time()
            latency = end_time - start_time
            logger.info("Tool run finished", extra={"run_id": run_id, "tool": self.name, "latency": latency, "output": output})
            metrics_tracker.record_tool_event(
                run_id=run_id, tool_name=self.name, source="vector_db", success=True
            )

            return output

        except Exception as e:
            end_time = time.time()
            latency = end_time - start_time
            logger.error("Tool run failed", extra={"run_id": run_id, "tool": self.name, "latency": latency, "error": str(e)})
            metrics_tracker.record_tool_event(
                run_id=run_id, tool_name=self.name, source="vector_db", success=False, error=True
            )
            return {"error": f"Search failed: {e}"}
