import pandas as pd
import os
import sys
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from backend.config import DISEASE_FAISS_DB, MAYO_CSV

# Load embedding model

class DISEASEINFOAGENT:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.df = pd.read_csv(MAYO_CSV)

        self.vectorstore = FAISS.load_local(DISEASE_FAISS_DB, self.embeddings, allow_dangerous_deserialization=True)

    def match(self, query):
        results = self.vectorstore.similarity_search(query, k=3)
        result = ""
        for r in results:
            print(f"Disease: {r.metadata['disease']}, Text: {r.page_content}")
            overview = self.df.loc[self.df["disease"] == r.metadata['disease'], "Overview"].values[0]
            result += f"disease name: {r.metadata['disease']}, disease overview: {overview} \n\n"


        disease_info = result
        return disease_info
    

































