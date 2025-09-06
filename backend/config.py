import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = DATA_DIR  # because your labels.json and test_symptom_cases.csv are directly inside data/
VECTOR_DIR = os.path.join(DATA_DIR, "Vector", "symptom_faiss_db")  # fixed path

# File paths
TEST_CASES_CSV = os.path.join(BASE_DIR, "data","mayo_diseases.csv")
LABELS_JSON = os.path.join(RAW_DIR, "labels.json")
MAYO_CSV = os.path.join(BASE_DIR, "data","updated_df.csv")  # since this is in notebook/
FAISS_INDEX = os.path.join(VECTOR_DIR, "index.faiss")
FAISS_META = os.path.join(VECTOR_DIR, "index.pkl")
DISEASE_FAISS_DB = os.path.join(DATA_DIR, "Vector", "disease_faiss_db")
SKIN_TRAIN = os.path.join(BASE_DIR,"skin-disease-datasaet","train_set")
SKIN_TEST = os.path.join(BASE_DIR,"skin-disease-datasaet","test_set")
    
TEST_SCORES_JSON = os.path.join(BASE_DIR,"data","test.json")
README_PATH = os.path.join(BASE_DIR, "README.md")