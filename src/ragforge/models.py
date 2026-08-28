from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


@dataclass(slots=True)
class Chunk:
    id: str
    document_id: str
    text: str
    embedding: list[float]
    chunk_index: int = 0
    source: str = "unknown"
    content_hash: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(slots=True)
class SearchHit:
    chunk: Chunk
    score: float
    dense_score: float = 0.0
    sparse_score: float = 0.0
    rerank_score: float | None = None


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class Citation(BaseModel):
    id: int
    document_id: str
    source: str
    chunk_index: int
    excerpt: str
    score: float


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    latency_ms: float
    retrieval_ms: float
    generation_ms: float


class IngestDocument(BaseModel):
    id: str
    text: str = Field(min_length=1)
    source: str = "api"
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    documents: list[IngestDocument] = Field(min_length=1, max_length=1000)


class IngestResponse(BaseModel):
    documents_seen: int
    documents_updated: int
    chunks_upserted: int
    documents_unchanged: int

