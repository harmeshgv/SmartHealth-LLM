from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from app.config import settings

if not settings.SUPABASE_DB_URL:
    raise RuntimeError("❌ SUPABASE_DB_URL is missing — check your .env")

engine = create_engine(
    settings.SUPABASE_DB_URL,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

Base = declarative_base()
