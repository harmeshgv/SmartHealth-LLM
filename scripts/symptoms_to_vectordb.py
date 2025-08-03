import sys
import os
import pandas as pd
from tqdm import tqdm

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document


# Setup system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.config import MAYO_CSV, VECTOR_DIR
from backend.utils.data_preprocessing import DataPreprocessing

class Symptoms_To_VectorDB:
    def __init__(self, csv_path=MAYO_CSV):
        self.preprocessor = DataPreprocessing()
        self.df = pd.read_csv(csv_path)
        self.embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


        
    def build_vector_db(self, save_path=VECTOR_DIR):
        tqdm.pandas(desc="Cleaning Symptoms")
        self.df["symptoms_main"] = self.df["Symptoms"].progress_apply(lambda x: self.preprocessor.preprocess(x))
        print("_"*30)
        print(self.df["symptoms_main"].head())
        documents = []

        for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc="📦 Building Vector DB"):
            word_list = row["symptoms_main"]  
            for word in word_list:
                documents.append(Document(page_content=word, metadata={"disease": row["disease"]}))

        # Generate embeddings and store
        vectorstore = FAISS.from_documents(documents, self.embedder)
        vectorstore.save_local(save_path)
        print(f"\n✅ Vector DB saved at: {save_path}")

if __name__ == "__main__":
    processor = Symptoms_To_VectorDB()
    processor.build_vector_db()
