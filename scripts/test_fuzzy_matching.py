import argparse
import logging
from fuzzy_match_tester_tool import FuzzyMatchTesterTool
from backend.config import settings

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_disease_matching(disease_name: str):
    """
    Initializes the FuzzyMatchTesterTool and executes a search for the given disease name.
    """
    try:
        tester = FuzzyMatchTesterTool(csv_path=settings.MAYO_CSV)
        matches = tester.execute(query=disease_name)

        if not matches:
            logging.warning(f"No matches found for '{disease_name}'")
            return

        logging.info(f"Found {len(matches)} potential match(es) for '{disease_name}':\n")
        # Print results in a table-like format
        print(f"{'Score':<10} | {'Matched Disease Name'}")
        print("-" * 40)
        for match in matches:
            print(f"{match['score']:<10} | {match['matched_disease']}")
        
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test fuzzy matching for a disease name against the database."
    )
    parser.add_argument(
        "disease_name",
        type=str,
        help="The name of the disease to search for (e.g., 'diabetis')."
    )

    args = parser.parse_args()
    test_disease_matching(args.disease_name)
