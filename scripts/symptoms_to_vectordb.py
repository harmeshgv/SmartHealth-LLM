import sys
import os
import pandas as pd
from tqdm import tqdm

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# Setup system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.utils.text_cleaning import Text_Preprocessing
from backend.utils.filtering_with_ner import RemoveUselessWords

class Symptoms_To_VectorDB:
    def __init__(self, csv_path="backend/data/mayo_diseases.csv"):
        self.text_preprocessing = Text_Preprocessing()
        self.remove = RemoveUselessWords()
        self.df = pd.read_csv(csv_path)
        self.embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    def clean_symptoms(self):
        tqdm.pandas(desc="🔍 Cleaning Symptoms")
        self.df["symptoms_cleaned"] = self.df["Symptoms"].progress_apply(lambda x: self.text_preprocessing.go_on(x))
        self.df["symptoms_main"] = self.df["symptoms_cleaned"].apply(lambda x: self.remove(x))

    def build_vector_db(self, save_path="Vector/symptom_faiss_db"):
        documents = []

        for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc="📦 Building Vector DB"):
            word_list = row["symptoms_main"].split()  # space-separated cleaned words
            for word in word_list:
                documents.append(Document(page_content=word, metadata={"disease": row["disease"]}))

        # Generate embeddings and store
        vectorstore = FAISS.from_documents(documents, self.embedder)
        vectorstore.save_local(save_path)
        print(f"\n✅ Vector DB saved at: {save_path}")

if __name__ == "__main__":
    processor = Symptoms_To_VectorDB()
    processor.clean_symptoms()
    processor.build_vector_db()
