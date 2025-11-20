import pandas as pd
import logging
from typing import List, Set

# Temporarily add backend to sys.path to allow direct import
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.tools.disease_info_retriever_tool import DiseaseInfoRetrieverTool
from backend.config import settings

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def get_unique_diseases_from_txt(file_path: str) -> Set[str]:
    """Loads unique disease names from the specified text file."""
    logging.info(f"Loading unique diseases from: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Read lines, strip whitespace, filter out empty lines, and convert to a set
            diseases = set(line.strip() for line in f if line.strip())
        
        logging.info(f"Found {len(diseases)} unique diseases in the text file.")
        return diseases
    except FileNotFoundError:
        logging.error(f"Unique diseases file not found at: {file_path}")
        raise
    except Exception as e:
        logging.error(f"Error processing unique diseases file: {e}")
        raise


def check_database_coverage(diseases_to_check: Set[str], retriever_tool: DiseaseInfoRetrieverTool) -> None:
    """Checks which diseases are present in the retriever tool's database."""
    found_count = 0
    not_found_diseases = []

    total_diseases = len(diseases_to_check)
    if total_diseases == 0:
        logging.warning("No diseases found to check.")
        return

    logging.info(f"Checking coverage for {total_diseases} diseases...")

    for i, disease in enumerate(diseases_to_check):
        result = retriever_tool.execute_sync(disease_name=disease)
        if "error" not in result:
            found_count += 1
        else:
            not_found_diseases.append(disease)
        
        # Log progress
        if (i + 1) % 100 == 0 or (i + 1) == total_diseases:
            logging.info(f"Checked {i+1}/{total_diseases} diseases...")
    
    logging.info("--- Database Coverage Report ---")
    
    coverage_percentage = (found_count / total_diseases) * 100
    
    print("\n" + "="*40)
    print("      Database Coverage Results")
    print("="*40)
    print(f"Total Unique Diseases Checked:    {total_diseases}")
    print(f"Diseases Found in Current DB:     {found_count}")
    print(f"Coverage Percentage:              {coverage_percentage:.2f}%")
    print("="*40 + "\n")

    if not_found_diseases:
        logging.warning(f"Found {len(not_found_diseases)} missing diseases. See list below:")
        # Sort for consistent output
        not_found_diseases.sort()
        for missing in not_found_diseases:
            print(f"  - {missing.title()}")
    else:
        logging.info("✅ Excellent! All diseases were found in the database.")


if __name__ == "__main__":
    # Add a synchronous execute method to the tool for easier scripting
    def execute_sync(self, **kwargs):
        import asyncio
        return asyncio.run(self.execute(**kwargs))
    
    DiseaseInfoRetrieverTool.execute_sync = execute_sync

    unique_diseases_path = "backend/data/unique_diseases.txt"
    
    try:
        # Initialize embeddings for the retriever tool
        from backend.utils.embeddings import get_embeddings
        embeddings_instance = get_embeddings()

        unique_diseases = get_unique_diseases_from_txt(unique_diseases_path)
        retriever_tool = DiseaseInfoRetrieverTool(csv_path=settings.MAYO_CSV, embeddings=embeddings_instance)
        check_database_coverage(unique_diseases, retriever_tool)
    except Exception as e:
        logging.error(f"Analysis failed: {e}")
