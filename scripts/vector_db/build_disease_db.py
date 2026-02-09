import sys
import os
import pandas as pd
from tqdm import tqdm
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

# Ensure the backend directory is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'backend')))
from app.config import settings

def build_and_save_disease_db(csv_path: str, db_output_path: str, embeddings: Embeddings):
    """
    Builds a FAISS vector store of disease names from a CSV file.

    Args:
        csv_path (str): The path to the input CSV file.
        db_output_path (str): The path to save the FAISS database.
        embeddings (Embeddings): The embeddings model to use.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return

    texts = []
    metadatas = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing CSV for disease names"):
        disease = row.get("disease")
        if disease:
            texts.append(disease)
            metadatas.append({"disease": disease})  # Keep track of disease name

    if not texts:
        print("No disease names to process. FAISS DB will not be created.")
        return
        
    print("Building FAISS vector store for disease names...")
    vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    
    os.makedirs(db_output_path, exist_ok=True)
    vectorstore.save_local(db_output_path)
    print(f"Disease name FAISS DB saved at {db_output_path}")

def main():
    """Main function to build the database from configuration."""
    from langchain_huggingface import HuggingFaceEmbeddings

    print("Loading embedding model for disease DB...")
    hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    build_and_save_disease_db(settings.MAYO_CSV, settings.DISEASE_INFO_FAISS_DB, hf_embeddings)

if __name__ == "__main__":
    main()
