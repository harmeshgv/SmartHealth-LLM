# disease_info_tool.py
import pandas as pd
from langchain.tools import tool
from langchain_community.vectorstores import FAISS
from backend.config import DISEASE_INFO_FAISS_DB, MAYO_CSV
from backend.utils.embeddings import get_embeddings  # your own helper


embeddings = get_embeddings()  # or HuggingFaceEmbeddings() if that’s your fn
df = pd.read_csv(MAYO_CSV)
vectorstore = FAISS.load_local(
    DISEASE_INFO_FAISS_DB, embeddings, allow_dangerous_deserialization=True
)


@tool
def match_disease_info(query: str) -> str:
    """
    Retrieve top-3 matching diseases from the FAISS DB
    and return a short overview for each.
    """
    results = vectorstore.similarity_search(query, k=3)

    if not results:
        return "No close matches found."

    out = []
    for r in results:
        disease = r.metadata.get("disease", "Unknown")
        # get Overview from the CSV
        overview_series = df.loc[df["disease"] == disease, "Overview"]
        symptoms_series = df.loc[df["disease"] == disease, "Symptoms"]

        overview = (
            overview_series.values[0]
            if not overview_series.empty
            else "Overview not found."
        )
        symptoms = (
            symptoms_series.values[0]
            if not symptoms_series.empty
            else "Symptom not found."
        )

        out.append(f"Disease: {disease}\nOverview: {overview}\nSymptoms: {symptoms}")

    return "\n\n".join(out)
