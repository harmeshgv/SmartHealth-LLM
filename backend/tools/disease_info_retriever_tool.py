# backend/tools/disease_info_retriever_tool.py
import logging
import csv
from typing import List, Dict, Any, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document

from .base import BaseTool

logger = logging.getLogger(__name__)

class DiseaseInfoRetrieverTool(BaseTool):
    """
    A tool to retrieve detailed information about diseases from a local CSV database.
    Uses exact, normalized, and semantic matching to find the disease.
    """

    name: str = "Disease Information Retriever"
    description: str = (
        "Retrieves information about a specific disease, such as overview, symptoms, "
        "causes, and treatments. Input should be a disease name."
    )
    db: List[Dict[str, str]] = []
    disease_list: List[str] = []
    name_vectorstore: FAISS
    db_map: Dict[str, Dict[str, str]] = {} # Declare as instance attribute type

    def __init__(self, csv_path: str, embeddings: Embeddings):
        super().__init__()
        self.csv_path = csv_path
        try:
            self.db = self._load_csv()
            self.disease_list = [row["disease"] for row in self.db]
            self.db_map = {row["disease"].lower().strip(): row for row in self.db} # Initialize db_map
            
            # Build a FAISS index from the disease names for semantic search
            name_docs = [Document(page_content=name) for name in self.disease_list]
            self.name_vectorstore = FAISS.from_documents(name_docs, embeddings)
            
            logger.info(f"DiseaseInfoRetriever loaded data from {csv_path}")
        except Exception as e:
            logger.error(f"Failed to load disease database from {csv_path}: {e}", exc_info=True)
            raise

    def _load_csv(self) -> List[Dict[str, str]]:
        """Loads the disease data from the CSV file."""
        with open(self.csv_path, mode="r", encoding="utf-8") as file:
            return list(csv.DictReader(file))

    def _find_best_match(self, disease_name: str, threshold: float = 0.85) -> Optional[str]:
        """Finds the best matching disease name using exact, normalized, and semantic search."""
        if not disease_name or not self.disease_list:
            return None

        disease_name_lower = disease_name.lower().strip()

        # 1. Exact match (case-insensitive)
        for disease in self.disease_list:
            if disease.lower() == disease_name_lower:
                logger.debug(f"Found exact match for '{disease_name}': '{disease}'")
                return disease

        # 2. Normalized match (replace hyphens/underscores)
        normalized_search = " ".join(disease_name_lower.replace("/", " ").replace("-", " ").replace("_", " ").split())
        for disease_key, disease_data in self.db_map.items():
            normalized_db_key = " ".join(disease_key.replace("/", " ").replace("-", " ").replace("_", " ").split())
            if normalized_db_key == normalized_search:
                logger.debug(f"Found normalized match for '{disease_name}': '{disease_data['disease']}'")
                return disease_key # Return the original key from db_map
        
        # 3. Semantic (vector) search
        logger.debug(f"No exact/normalized match for '{disease_name}'. Trying semantic search.")
        # Ensure the query passed to the vectorstore is the same one used for assertions
        results = self.name_vectorstore.similarity_search_with_score(disease_name_lower, k=1, score_threshold=threshold)
        if results:
            best_match_doc, score = results[0]
            match_key = best_match_doc.page_content.lower().strip()
            logger.info(f"Found semantic match for '{disease_name}': '{match_key}' with score {score:.2f}")
            return match_key

        logger.warning(f"No suitable match found for disease '{disease_name}'")
        return None

    async def execute(self, disease_name: str, fields: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Retrieves information for a given disease name.

        Args:
            disease_name: The name of the disease to search for.
            fields: An optional list of fields to return. If None, all fields are returned.

        Returns:
            A dictionary containing the disease information or an error.
        """
        if not disease_name:
            return {"error": "No disease name provided."}

        matched_disease_key = self._find_best_match(disease_name)

        if not matched_disease_key:
            return {"error": f"Could not find a match for '{disease_name}' in the database."}
        
        # Use the case-insensitive map for the final lookup
        info = self.db_map.get(matched_disease_key, {})
        if not info:
             return {"error": f"Data integrity error: Matched disease key '{matched_disease_key}' not found in map."}

        if fields:
            result = {field: info.get(field, "N/A") for field in fields}
        else:
            result = dict(info) # Return all fields

        logger.info(f"Successfully retrieved info for disease: '{info['disease']}' (searched for: '{disease_name}')")
        return {"info": result}

# testing
if __name__ == "__main__":
    import asyncio
    # Import necessary components for local testing
    from backend.utils.embeddings import get_embeddings # Import get_embeddings
    
    embeddings = get_embeddings() # Use the singleton get_embeddings function    
    D=DiseaseInfoRetrieverTool(csv_path="backend/data/updated_df.csv", embeddings=embeddings)
    result = asyncio.run(D.execute("dengue"))
    print(result)