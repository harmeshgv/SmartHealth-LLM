import csv
import os
from typing import List, Dict, Optional, Any
from app.tools.base_tool import BaseTool
from app.utils.embeddings import EmbeddingSingleton
import faiss
from app.config import settings


class DiseaseInfoRetrieverTool(BaseTool):
    """
    Retrieves disease info from local CSV using:
    - exact match
    - normalized match
    - semantic similarity (FAISS)
    """

    name = "disease_info_retriever"
    description = "Returns detailed disease info from the local CSV database."

    def __init__(self):
        super().__init__()

        self.csv_path = settings.DISEASE_INFO_PATH
        self.faiss_path = os.path.join(settings.FAISS_DISEASE_PATH, "index.faiss")
        self.pkl_path = os.path.join(settings.FAISS_DISEASE_PATH, "index.pkl")

        # Load embeddings
        self.embedding_model = EmbeddingSingleton.get_instance()

        # Load CSV database
        self.db = self._load_csv()
        self.disease_list = [row["disease"].strip() for row in self.db]
        self.db_map = {row["disease"].lower().strip(): row for row in self.db}

        # Load FAISS index
        self.index = faiss.read_index(self.faiss_path)

    # -------------------------------
    def _load_csv(self) -> List[Dict[str, str]]:
        with open(self.csv_path, mode="r", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    # -------------------------------
    def _normalize(self, x: str) -> str:
        return " ".join(x.replace("-", " ").replace("_", " ").split())

    # -------------------------------
    def _find_best_match(self, disease_name: str, threshold: float = 0.80) -> Optional[str]:
        query = disease_name.lower().strip()

        # 1. exact match
        if query in self.db_map:
            return query

        # 2. normalized match
        for key in self.db_map:
            if self._normalize(key) == self._normalize(query):
                return key

        # 3. partial token containment ("dengue" ⊆ "dengue fever")
        query_tokens = set(query.split())
        for key in self.db_map:
            key_tokens = set(key.split())
            if query_tokens.issubset(key_tokens):
                return key

        # 4. semantic match
        emb = self.embedding_model.embed_query(disease_name).reshape(1, -1)
        scores, idx = self.index.search(emb, 1)

        if scores[0][0] < threshold:
            return None

        return self.disease_list[idx[0][0]].lower()


    # -------------------------------
    async def run(self, disease_name: str,
                  fields: Optional[List[str]] = None) -> Dict[str, Any]:

        if not disease_name:
            return {"error": "No disease name provided"}

        match_key = self._find_best_match(disease_name)

        if not match_key:
            return {"error": f"No match found for {disease_name}"}

        info = self.db_map.get(match_key, {})

        # If user wants specific fields
        if fields:
            return {"info": {f: info.get(f, "N/A") for f in fields}}

        # Return entire info row
        return {"info": info}
