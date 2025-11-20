import logging
from typing import List, Dict, Any
import csv
from rapidfuzz import process, utils, fuzz

# Temporarily add backend to sys.path to allow direct import
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.config import settings

logger = logging.getLogger(__name__)

class FuzzyMatchTesterTool:
    """
    A diagnostic tool to test fuzzy matching recall against the disease database.
    """
    db: List[Dict[str, str]] = []
    disease_list: List[str] = []

    def __init__(self, csv_path: str):
        try:
            self.db = self._load_csv(csv_path)
            self.disease_list = [row["disease"] for row in self.db]
            logger.info(f"Fuzzy Match Tester loaded {len(self.disease_list)} diseases from {csv_path}")
        except Exception as e:
            logger.error(f"Failed to load disease DB for testing: {e}", exc_info=True)
            raise

    def _load_csv(self, file_path: str) -> List[Dict[str, str]]:
        """Loads data from the specified CSV file."""
        with open(file_path, mode='r', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)
            return [row for row in reader]

    def execute(self, query: str, score_cutoff: int = 80) -> List[Dict[str, Any]]:
        """
        Finds all potential disease names in the DB that fuzzy match the query.

        Args:
            query: The disease name to search for.
            score_cutoff: The minimum similarity score (0-100) to consider a match.

        Returns:
            A list of dictionaries, each containing the matched disease and its score.
        """
        if not isinstance(query, str) or not query.strip():
            logger.error("Query must be a non-empty string.")
            return []
        
        normalized_query = utils.default_process(query)
        
        try:
            matches = process.extract(
                normalized_query,
                self.disease_list,
                scorer=fuzz.WRatio, # Corrected scorer
                score_cutoff=score_cutoff,
                limit=10 # Limit to top 10 matches
            )
            
            if not matches:
                return []

            # Format the results
            formatted_matches = [
                {"matched_disease": match[0], "score": f"{match[1]:.2f}", "original_db_entry": self.disease_list[match[2]]}
                for match in matches
            ]
            return formatted_matches
        except Exception as e:
            logger.error(f"An error occurred during fuzzy matching: {e}", exc_info=True)
            return []
