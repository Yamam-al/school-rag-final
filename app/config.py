# app/config.py
from __future__ import annotations
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator
from pathlib import Path


class Settings(BaseSettings):
    """
    Globale App-Einstellungen.

    Pflichtwerte kommen aus .env.
    EMBEDDING_MODEL muss auf einen lokalen Sentence-Transformer Snapshot zeigen
    (Ordner mit config.json).
    """
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM
    LLM_BASE_URL: str
    LLM_API_KEY: str
    LLM_MODEL: str

    # Embeddings (lokaler Snapshot, Ordner mit config.json)
    EMBEDDING_MODEL: str

    # Chroma
    CHROMA_PATH: str = "data/chroma"

    # Dateiablagen
    DATA_SOURCE_DIR: str = "data/source"
    MODEL_CACHE: str = "data/cache"

    # Retrieval/Chunking
    CHUNK_SIZE: int = 900
    CHUNK_OVERLAP: int = 120
    TOP_K: int = 5

    @field_validator("EMBEDDING_MODEL", mode="after")
    def _check_embedder_path(cls, v: str) -> str:
        p = Path(v)
        if not p.exists():
            raise ValueError(f"EMBEDDING_MODEL path not found: {p}")
        if not (p / "config.json").exists():
            raise ValueError("EMBEDDING_MODEL must point to a snapshot folder containing config.json")
        return str(p)

    @field_validator("CHROMA_PATH", "MODEL_CACHE", "DATA_SOURCE_DIR", mode="after")
    def _ensure_dirs(cls, v: str) -> str:
        Path(v).mkdir(parents=True, exist_ok=True)
        return v


settings = Settings()
