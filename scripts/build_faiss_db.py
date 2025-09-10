import sys
import os
import pandas as pd
from tqdm import tqdm
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.config import MAYO_CSV, SYMPTOM_FAISS_DB

# Example data
disease_symptoms = {}

df = pd.read_csv(MAYO_CSV)

for _, row in tqdm(df.iterrows(), total=len(df)):
    disease = row["disease"]
    symptoms = row["updated"]
    symptoms = symptoms.split(",")
    disease_symptoms[disease] = symptoms


texts = ["; ".join(symptoms) for symptoms in disease_symptoms.values()]
metadatas = [{"disease": name} for name in disease_symptoms.keys()]

# Use HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Build FAISS vector store
vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
vectorstore.save_local(SYMPTOM_FAISS_DB)
print(f"FAISS DB saved at {SYMPTOM_FAISS_DB}")
