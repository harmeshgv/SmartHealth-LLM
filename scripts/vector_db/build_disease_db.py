import sys
import os
import pandas as pd
from tqdm import tqdm
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.config import MAYO_CSV, DISEASE_INFO_FAISS_DB

# Load CSV
df = pd.read_csv(MAYO_CSV)

texts = []
metadatas = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    disease = row["disease"]
    texts.append(disease)
    metadatas.append({"disease": disease})  # keep track of disease name

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Build FAISS vector store
vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
vectorstore.save_local(DISEASE_INFO_FAISS_DB)
print(f"FAISS DB saved at {DISEASE_INFO_FAISS_DB}")
