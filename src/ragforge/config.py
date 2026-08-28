from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAGFORGE_", env_file=".env", extra="ignore")

    backend: Literal["memory", "postgres"] = "memory"
    database_url: str = "postgresql://ragforge:ragforge@localhost:5432/ragforge"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    embedding_dimensions: int = 384
    candidate_k: int = 20
    top_k: int = 5
    openai_api_key: str | None = None
    openai_model: str = "gpt-4.1-mini"


@lru_cache
def get_settings() -> Settings:
    return Settings()

