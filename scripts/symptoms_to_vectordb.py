import pandas as pd
from tqdm import tqdm

from langchain_huggingface import HuggingFaceEmbeddings

class SymptomDiseaseDB:
    def __init__(self, csv_path, symptom_extractor, embedder_model="all-MiniLM-L6-v2"):
        self.df = pd.read_csv(csv_path)
        self.extractor = symptom_extractor
        self.embedder = HuggingFaceEmbeddings(model_name=embedder_model)
        self.disease_db = {}

    def build(self):
        for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc="🔍 Processing diseases"):
            disease = row["disease"]
            symptoms_text = row["Symptoms"]
            if not isinstance(symptoms_text, str):
                continue
            symptom_list = self.extractor.extract(symptoms_text)
            if not symptom_list:
                continue
            symptom_embeddings = self.embedder.embed_documents(symptom_list)
            self.disease_db[disease] = {
                "symptoms": symptom_list,
                "embeddings": symptom_embeddings
            }

    def get_db(self):
        return self.disease_db
