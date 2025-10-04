# backend/tools/disease_matcher_tool.py

from typing import List, Dict

from langchain_community.vectorstores import FAISS
from langchain.tools import tool

from backend.config import SYMPTOM_FAISS_DB
from backend.utils.embeddings import get_embeddings


class SymptomDiseaseMatcher:
    def __init__(self, db_path=SYMPTOM_FAISS_DB):

        self.embeddings = get_embeddings()

        self.vectorstore = FAISS.load_local(
            db_path, self.embeddings, allow_dangerous_deserialization=True
        )

    def match(self, query: List[str], k: int = 3) -> Dict[str, str]:
        query = ",".join(query)
        results = self.vectorstore.similarity_search(query, k=3)
        matches = []
        print(results)
        print("-" * 40)

        for r in results:
            matches.append(
                {"disease": r.metadata["disease"], "symptoms": r.page_content}
            )
        return matches


matcher = SymptomDiseaseMatcher()


@tool
def match_disease_from_symptom(query: str) -> str:
    """
    Matches symptoms to diseases using FAISS.

    Input:
        A comma-separated string of symptoms (e.g., "fever, cough, sore throat").

    Output:
        A JSON-like string containing matched diseases and their details.
    """
    symptoms = [s.strip() for s in query.split(",")]
    matches = matcher.match(symptoms)
    if not matches:
        return "No matches found."

    response = {"input_symptoms": symptoms, "matched_diseases": matches}
    return str(response)


if __name__ == "__main__":
    result = match_disease_from_symptom.invoke({"query": "fever, cough, headache"})
    print(result)
