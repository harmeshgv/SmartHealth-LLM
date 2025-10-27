# backend/tools/disease_matcher_tool.py

from typing import List, Dict, Any
from langchain_community.vectorstores import FAISS

from config import SYMPTOM_FAISS_DB
from utils.embeddings import get_embeddings


class SymptomDiseaseMatcher:
    def __init__(self, db_path=SYMPTOM_FAISS_DB):
        self.embeddings = get_embeddings()
        self.vectorstore = FAISS.load_local(
            db_path, self.embeddings, allow_dangerous_deserialization=True
        )

    def match(self, query: List[str], k: int = 3) -> List[Dict[str, Any]]:
        query_str = ",".join(query)
        results = self.vectorstore.similarity_search(query_str, k=k)
        matches = []

        print("DEBUG - FAISS Results:")
        print(results)
        print("-" * 40)

        for r in results:
            matches.append(
                {
                    "disease": r.metadata.get("disease", "Unknown"),
                    "symptoms": r.page_content,
                    "score": getattr(r, "score", 0.0),  # Similarity score if available
                }
            )
        return matches


# Instantiate the matcher
matcher = SymptomDiseaseMatcher()


# LangGraph-ready function
def match_disease_tool(input_dict: Dict) -> Dict:
    """
    LangGraph-compatible function for symptom-disease matching.

    Args:
        input_dict: {"symptoms": "fever, cough"} or {"symptoms": ["fever", "cough"]}

    Returns:
        Dict with matching results
    """
    if not input_dict:
        return {"error": "No input provided"}

    symptoms_input = input_dict.get("symptoms", "")

    if isinstance(symptoms_input, list):
        symptoms = symptoms_input
    else:
        symptoms = [s.strip() for s in symptoms_input.split(",") if s.strip()]

    try:
        matches = matcher.match(symptoms)

        return {
            "input_symptoms": symptoms,
            "matched_diseases": matches,
            "match_count": len(matches),
            "status": "success" if matches else "no_matches_found",
        }
    except Exception as e:
        return {
            "input_symptoms": symptoms,
            "matched_diseases": [],
            "match_count": 0,
            "status": "error",
            "error": str(e),
        }


if __name__ == "__main__":
    # Test the tool
    print("Testing Disease Matcher Tool:")
    print("=" * 50)

    # Test with LangGraph function
    result = match_disease_tool({"symptoms": "fever, cough, headache"})
    print("LangGraph function result:")
    print(result)
