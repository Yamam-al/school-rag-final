# app/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator
from pathlib import Path

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM (Pflicht, aus .env)
    LLM_BASE_URL: str
    LLM_API_KEY: str
    LLM_MODEL: str

    # Embeddings (Pflicht, aus .env)  muss auf Snapshot-Ordner mit config.json zeigen
    EMBEDDING_MODEL: str

    # Caches und Speicher
    MODEL_CACHE: str = "./models"
    CHROMA_PATH: str

    # RAG
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 120
    TOP_K: int = 4

    @field_validator("EMBEDDING_MODEL", mode="after")
    def _check_embedder_path(cls, v: str) -> str:
        p = Path(v)
        if not p.exists():
            raise ValueError(f"EMBEDDING_MODEL path not found: {p}")
        if not (p / "config.json").exists():
            raise ValueError("EMBEDDING_MODEL must point to a snapshot folder containing config.json")
        return str(p)

    @field_validator("CHROMA_PATH", "MODEL_CACHE", mode="after")
    def _ensure_dirs(cls, v: str) -> str:
        Path(v).mkdir(parents=True, exist_ok=True)
        return v

settings = Settings()
# Arbeitsordner für Uploads sicherstellen
Path("data/source").mkdir(parents=True, exist_ok=True)
