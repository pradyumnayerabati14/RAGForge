from typing import Protocol

from ragforge.embeddings import Embedder
from ragforge.models import SearchHit
from ragforge.store import ChunkStore


class Reranker(Protocol):
    def rerank(self, query: str, hits: list[SearchHit], limit: int) -> list[SearchHit]: ...


class LexicalReranker:
    def rerank(self, query: str, hits: list[SearchHit], limit: int) -> list[SearchHit]:
        terms = set(query.lower().split())
        for hit in hits:
            overlap = len(terms & set(hit.chunk.text.lower().split())) / max(len(terms), 1)
            hit.rerank_score = 0.7 * hit.score + 0.3 * overlap
        return sorted(hits, key=lambda h: h.rerank_score or 0, reverse=True)[:limit]


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError("Install ragforge[ml] for cross-encoder reranking") from exc
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, hits: list[SearchHit], limit: int) -> list[SearchHit]:
        scores = self.model.predict([(query, h.chunk.text) for h in hits])
        for hit, score in zip(hits, scores, strict=True):
            hit.rerank_score = float(score)
        return sorted(hits, key=lambda h: h.rerank_score or 0, reverse=True)[:limit]


class HybridRetriever:
    def __init__(self, store: ChunkStore, embedder: Embedder, reranker: Reranker,
                 candidate_k: int = 20, rrf_k: int = 60):
        self.store, self.embedder, self.reranker = store, embedder, reranker
        self.candidate_k, self.rrf_k = candidate_k, rrf_k

    def search(self, query: str, limit: int = 5) -> list[SearchHit]:
        vector = self.embedder.embed([query])[0]
        dense = self.store.dense_search(vector, self.candidate_k)
        sparse = self.store.sparse_search(query, self.candidate_k)
        merged: dict[str, SearchHit] = {}
        for results, field in ((dense, "dense_score"), (sparse, "sparse_score")):
            for rank, hit in enumerate(results, 1):
                current = merged.setdefault(hit.chunk.id, SearchHit(hit.chunk, 0.0))
                current.score += 1 / (self.rrf_k + rank)
                setattr(current, field, getattr(hit, field))
        return self.reranker.rerank(query, list(merged.values()), limit)

