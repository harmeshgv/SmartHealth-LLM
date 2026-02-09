import sys
import os
import pandas as pd
from tqdm import tqdm
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

# Ensure the backend directory is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'backend')))
from app.config import settings

def build_and_save_faiss_db(csv_path: str, db_output_path: str, embeddings: Embeddings):
    """
    Builds a FAISS vector store from a CSV file and saves it to a specified path.

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

    disease_symptoms = {}
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing CSV"):
        disease = row.get("disease")
        symptoms_str = row.get("updated", "")
        if disease and symptoms_str:
            symptoms = symptoms_str.split(",")
            disease_symptoms[disease] = symptoms

    texts = ["; ".join(symptoms) for symptoms in disease_symptoms.values()]
    metadatas = [{"disease": name} for name in disease_symptoms.keys()]

    if not texts:
        print("No text data to process. FAISS DB will not be created.")
        return

    # Build FAISS vector store
    print("Building FAISS vector store...")
    vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    
    # Save the vector store
    os.makedirs(db_output_path, exist_ok=True)
    vectorstore.save_local(db_output_path)
    print(f"FAISS DB saved at {db_output_path}")

def main():
    """Main function to build the database from configuration."""
    from langchain_huggingface import HuggingFaceEmbeddings
    
    # Use HuggingFace embeddings
    print("Loading embedding model...")
    hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Build and save the database
    build_and_save_faiss_db(settings.MAYO_CSV, settings.SYMPTOM_FAISS_DB, hf_embeddings)

if __name__ == "__main__":
    main()
