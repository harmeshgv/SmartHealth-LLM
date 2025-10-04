# backend/tools/disease_info_retriever.py

import csv
import json
from typing import List, Dict
from langchain.tools import tool
from backend.config import MAYO_CSV
from rapidfuzz import process


class DiseaseInfoRetriever:
    def __init__(self, csv_path: str = MAYO_CSV):
        self.csv_path = csv_path
        self.df = self.load_csv()

    def load_csv(self) -> List[Dict]:
        with open(self.csv_path, mode="r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            data = [row for row in reader]
        return data

    def best_match(self, disease_name: str, disease_list: List[str], threshold=0.80):
        result = process.extractOne(disease_name, disease_list)
        match = getattr(result, "value", result[0])
        score = getattr(result, "score", result[1])
        if score >= threshold * 100:
            return match
        return None

    def get_info(self, disease_name: str, fields: list) -> Dict:
        disease_list = [row["disease"] for row in self.df]
        best_disease = self.best_match(disease_name, disease_list)
        if not best_disease:
            return {
                "error": "Disease name didn't match any known disease in local database"
            }

        for row in self.df:
            if row["disease"] == best_disease:
                return {field: row.get(field, "N/A") for field in fields}
        return {"error": "Disease not found."}


retriever = DiseaseInfoRetriever()

# all possible fields
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


@tool
def retrieve_disease_info(disease_name: str) -> str:
    """
    Retrieves full disease information from the local CSV.

    Input:
        A string containing the disease name.

    Output:
        A JSON-like string containing all fields of the disease information.
    """
    info = retriever.get_info(disease_name.strip(), all_fields)
    response = {"input_disease_name": disease_name, "info": info}
    return str(response)


if __name__ == "__main__":
    # quick test
    print(retrieve_disease_info("HIV"))
