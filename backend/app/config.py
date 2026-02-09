from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from typing import ClassVar, Dict, Tuple


class Settings(BaseSettings):

    # ---------- CONFIG RULES ----------
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # ignore unknown env vars (fixes your error)
    )

    # ---------- GENERAL ----------
    ENV: str = "development"
    DEBUG: bool = True
    APP_NAME: str = "MultiAgent Backend"

    # ---------- LLM KEYS ----------
    GROQ_API_KEY: str | None = None

    # ---------- WEB SEARCH ----------
    SERPER_API_KEY: str | None = None

    # ---------- LLM HOSTS ----------
    OLLAMA_HOST: str = "http://localhost:11434"

    # ---------- LOCAL DB -----------
    DISEASE_INFO_PATH: str = "app/data/disease_db.csv"

    # ---------- VECTOR DB ----------
    FAISS_SYMPTOM_PATH: str = "app/data/vector/symptom_faiss_db/"
    FAISS_DISEASE_PATH: str = "app/data/vector/disease_faiss_db/"

    # ---------- ML MODELS ----------
    SKIN_MODEL_PATH: str = "app/models/skin_disease_model.pth"
    SKIN_MODEL_CLASS_PATH: str = "app/models/labels.json"

    # ---------- AGENT MODEL MAP ----------
    AGENT_MODEL_MAP: ClassVar[Dict[str, Tuple[str, str]]] = {
        "decider_agent": ("ollama", "llama3.2:latest"),
        "planner_agent": ("ollama", "llama3.2:latest"),
        "reasoning_agent": ("ollama", "llama3.2:latest"),
        "symptom_matcher_agent": ("ollama", "llama3.2:latest"),
        "disease_info_agent": ("ollama", "llama3.2:latest"),
        "image_agent": ("ollama", "llama3.2:latest"),
        "conversation_agent": ("ollama", "llama3.2:latest"),
    }


@lru_cache()
def get_settings():
    return Settings()


settings = get_settings()
