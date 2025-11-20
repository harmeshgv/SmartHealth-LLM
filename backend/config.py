from pydantic_settings import BaseSettings
from typing import List, Dict
import os
import json

def load_from_json(file_path: str) -> List | Dict:
    """Load data from a JSON file."""
    with open(file_path) as f:
        return json.load(f)

class Settings(BaseSettings):
    BASE_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    DATA_DIR: str = os.path.join(BASE_DIR, "data")
    DISEASE_PREDICTION_MODEL: str = os.path.join(BASE_DIR, "models", "trained_densenet.pth")
    SYMPTOM_FAISS_DB: str = os.path.join(DATA_DIR, "Vector", "symptom_faiss_db")
    DISEASE_INFO_FAISS_DB: str = os.path.join(DATA_DIR, "Vector", "disease_faiss_db")
    MAYO_CSV: str = os.path.join(DATA_DIR, "updated_df.csv")

    # --- LLM & Search Settings ---
    TEST_API_KEY: str
    TEST_API_BASE: str
    TEST_MODEL: str
    SERPER_API_KEY: str

    # --- App & Server Settings ---
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "app.log"
    UVICORN_HOST: str = "0.0.0.0"
    UVICORN_PORT: int = 8000
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    SESSION_EXPIRE_SECONDS: int = 3600  # 1 hour
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://harmeshgv.github.io/SmartHealth-LLM/",
    ]

    # --- Model & Business Logic Settings ---
    SKIN_DISEASE_CLASS_NAMES: List[str] = load_from_json(os.path.join(DATA_DIR, "labels.json"))
    SYMPTOM_MAPPING: Dict[str, str] = {
        "fever and cough": "Common cold",
        "headache and nausea": "Migraine",
        "sore throat and runny nose": "Common cold",
        "fever": "Influenza",
        "cough": "Bronchitis",
    }
    BIOMEDICAL_NER_MODEL_NAME: str = "d4data/biomedical-ner-all"
    
    class Config:
        # This line tells Pydantic to load variables from a file named .env
        env_file = "backend/.env"
        env_file_encoding='utf-8'

settings = Settings()