# backend/tools/disease_info_retriever.py

import csv
from typing import List, Dict
from rapidfuzz import process
import logging
from config import MAYO_CSV


class DiseaseInfoRetriever:
    def __init__(self, csv_path: str = MAYO_CSV):
        self.csv_path = csv_path
        self.df = self.load_csv()
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"DiseaseInfoRetriever loaded CSV: {csv_path}")

    def load_csv(self) -> List[Dict]:
        with open(self.csv_path, mode="r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            data = [row for row in reader]
        return data

    def best_match(self, disease_name: str, disease_list: List[str], threshold=0.70):
        if not disease_name or not disease_list:
            return None

        # Clean the disease name
        disease_name = disease_name.lower().strip()

        print(f"DEBUG - Cleaned search term: '{disease_name}'")

        # First, try exact match (case insensitive)
        for disease in disease_list:
            if disease.lower() == disease_name:
                print(f"DEBUG - Exact match found: '{disease}'")
                return disease

        # Then try normalized matching (remove special chars, spaces, etc.)
        normalized_search = (
            disease_name.replace("/", " ")
            .replace("-", " ")
            .replace("_", " ")
            .replace(" and ", " ")
            .replace("&", " ")
        )
        normalized_search = " ".join(normalized_search.split())  # Remove extra spaces

        for disease in disease_list:
            normalized_db = (
                disease.lower()
                .replace("/", " ")
                .replace("-", " ")
                .replace("_", " ")
                .replace(" and ", " ")
                .replace("&", " ")
            )
            normalized_db = " ".join(normalized_db.split())

            # Check if search term is contained in DB term or vice versa
            if normalized_search in normalized_db or normalized_db in normalized_search:
                print(f"DEBUG - Normalized match found: '{disease}'")
                return disease

            # Check for word overlap
            search_words = set(normalized_search.split())
            db_words = set(normalized_db.split())
            if search_words & db_words:  # If there's any word overlap
                overlap = search_words & db_words
                print(
                    f"DEBUG - Word overlap: {overlap} between '{disease_name}' and '{disease}'"
                )
                if len(overlap) >= 1:  # At least one word matches
                    return disease

        # Finally, use fuzzy matching
        result = process.extractOne(
            disease_name, disease_list, score_cutoff=threshold * 100
        )
        if result:
            match, score, _ = result
            print(f"DEBUG - Fuzzy match: '{match}' with score: {score}")
            return match

        print(f"DEBUG - No match found for '{disease_name}'")
        return None

    def get_info(self, disease_name: str, fields: list) -> Dict:
        disease_list = [row["disease"] for row in self.df]

        # Debug: print available diseases and the search term
        print(f"DEBUG - Searching for: '{disease_name}'")
        print(
            f"DEBUG - Available diseases: {disease_list[:10]}..."
        )  # First 10 diseases
        print(f"DEBUG - Total diseases in DB: {len(disease_list)}")

        best_disease = self.best_match(disease_name, disease_list)
        print(f"DEBUG - Best match: '{best_disease}'")

        if not best_disease:
            return {
                "error": "Disease name didn't match any known disease in local database"
            }

        for row in self.df:
            if row["disease"] == best_disease:
                result = {field: row.get(field, "N/A") for field in fields}
                print(f"DEBUG - Found info for: {best_disease}")
                return result

        return {"error": "Disease not found."}


# Instantiate retriever
retriever = DiseaseInfoRetriever()

# All possible fields
all_fields = [
    "disease",
    "Overview",
    "Symptoms",
    "When to see a doctor",
    "Causes",
    "Risk factors",
    "Complications",
    "Prevention",
    "Diagnosis",
    "Treatment",
    "Coping and support",
    "Preparing for your appointment",
    "Lifestyle and home remedies",
]


# LangGraph-ready function
def retrieve_disease_info(input_dict: Dict) -> Dict:
    """
    Retrieves full disease information from the local CSV.

    Args:
        input_dict (dict): {"disease": disease_name}

    Returns:
        dict: {
            "input_disease_name": str,
            "info": dict of all fields or error message
        }
    """
    if input_dict is None:
        return None
    disease_name = input_dict.get("disease", "")
    info = retriever.get_info(disease_name, all_fields)
    return {"input_disease_name": disease_name, "info": info}


# Quick test
if __name__ == "__main__":
    class_names = [
        "cellulitis",
        "impetigo",
        "athlete-foot",
        "nail-fungus",
        "ringworm",
        "cutaneous-larva-migrans",
        "chickenpox",
        "shingles",
    ]

    test_output = retrieve_disease_info({"disease": class_names[7]})
    import json

    print(json.dumps(test_output, indent=2))
