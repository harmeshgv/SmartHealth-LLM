import spacy
import pickle
import pandas as pd
from tqdm import tqdm
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.config import MAYO_CSV, VECTOR_DIR
# Constants
CSV_PATH = MAYO_CSV
SPACY_MODEL = "en_ner_bc5cdr_md"
EMBED_MODEL = "all-MiniLM-L6-v2"
OUTPUT_PATH = VECTOR_DIR + "disease_db.pkl"

def extract_symptoms_spacy(text, nlp):
    doc = nlp(text)
    return list({ent.text.lower().strip() for ent in doc.ents})

def build_disease_db():
    print("🔍 Loading data and models...")
    df = pd.read_csv(CSV_PATH)
    nlp = spacy.load(SPACY_MODEL)
    embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    disease_db = {}
    print("⚙️ Building disease symptom embeddings...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        disease = row["disease"]
        symptoms_text = row["Symptoms"]
        if not isinstance(symptoms_text, str) or not symptoms_text.strip():
            continue

        symptoms = extract_symptoms_spacy(symptoms_text, nlp)
        if not symptoms:
            continue

        embeddings = embedder.embed_documents(symptoms)
        disease_db[disease] = {
            "symptoms": symptoms,
            "embeddings": embeddings
        }

    print(f"✅ Processed {len(disease_db)} diseases.")
    print(f"💾 Saving to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(disease_db, f)
    print("🎉 Done.")

if __name__ == "__main__":
    build_disease_db()
