from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from typing import ClassVar, Dict, Tuple


class Settings(BaseSettings):

    # ---------- CONFIG RULES ----------
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"   # ignore unknown env vars (fixes your error)
    )

    # ---------- GENERAL ----------
    ENV: str = "development"
    DEBUG: bool = True
    APP_NAME: str = "MultiAgent Backend"

    # ---------- LLM KEYS ----------
    GROQ_API_KEY: str | None = None
    HF_API_KEY: str | None = None

    # ---------- WEB SEARCH ----------
    SERPER_API_KEY: str | None = None

    # ---------- LLM HOSTS ----------
    OLLAMA_HOST: str = "http://localhost:11434"
    VLLM_HOST: str = "http://localhost:8000"

    # ---------- MEMORY ----------
    REDIS_URL: str = "redis://localhost:6379/0"

    # ---------- DATABASE ----------
    POSTGRES_URL: str = "postgresql://user:password@localhost:5432/db"

    # ---------- SUPABASE ----------
    SUPABASE_URL: str | None = None
    SUPABASE_ANON_KEY: str | None = None
    SUPABASE_SERVICE_ROLE_KEY: str | None = None
    SUPABASE_JWT_SECRET: str | None = None

    # SQLAlchemy-ready DB URL (with SSL)
    SUPABASE_DB_URL: str | None = None

    JWT_EXPIRE_MINUTES: int = 60

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
        "decider_agent": ("groq", "llama-3.1-8b-instant"),
        "planner_agent": ("groq", "llama-3.3-70b-versatile"),
        "reasoning_agent": ("groq", "openai/gpt-oss-120b"),
        "symptom_matcher_agent": ("ollama", "llama3.2:latest"),
        "disease_info_agent": ("groq", "Llama-3.3-70b-versatile"),
        "image_agent": ("groq", "llama-3.2-11b-vision-instruct"),
        "conversation_agent": ("ollama", "phi3:latest"),
    }



@lru_cache()
def get_settings():
    return Settings()


settings = get_settings()
