import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")

DISEASE_PREDICTION_MODEL = os.path.join(BASE_DIR, "models", "trained_densenet.pth")

SYMPTOM_FAISS_DB = os.path.join(DATA_DIR, "Vector", "symptom_faiss_db")
DISEASE_INFO_FAISS_DB = os.path.join(DATA_DIR, "Vector", "disease_faiss_db")


MAYO_CSV = os.path.join(DATA_DIR, "updated_df.csv")
