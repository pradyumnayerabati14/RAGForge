import math
import re
from collections import Counter
from typing import Protocol

from ragforge.models import Chunk, SearchHit


class ChunkStore(Protocol):
    def upsert_document(self, chunks: list[Chunk]) -> int: ...
    def document_hashes(self, document_id: str) -> set[str]: ...
    def all_chunks(self) -> list[Chunk]: ...
    def dense_search(self, embedding: list[float], limit: int) -> list[SearchHit]: ...
    def sparse_search(self, query: str, limit: int) -> list[SearchHit]: ...


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class InMemoryChunkStore:
    def __init__(self) -> None:
        self._chunks: dict[str, Chunk] = {}

    def upsert_document(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        document_id = chunks[0].document_id
        self._chunks = {
            key: val for key, val in self._chunks.items() if val.document_id != document_id
        }
        self._chunks.update({chunk.id: chunk for chunk in chunks})
        return len(chunks)

    def document_hashes(self, document_id: str) -> set[str]:
        return {c.content_hash for c in self._chunks.values() if c.document_id == document_id}

    def all_chunks(self) -> list[Chunk]:
        return list(self._chunks.values())

    def dense_search(self, embedding: list[float], limit: int) -> list[SearchHit]:
        hits = []
        for chunk in self._chunks.values():
            score = sum(a * b for a, b in zip(embedding, chunk.embedding, strict=False))
            hits.append(SearchHit(chunk=chunk, score=score, dense_score=score))
        return sorted(hits, key=lambda hit: hit.score, reverse=True)[:limit]

    def sparse_search(self, query: str, limit: int) -> list[SearchHit]:
        query_terms = Counter(_tokens(query))
        chunks = self.all_chunks()
        if not chunks:
            return []
        doc_freq = Counter(term for chunk in chunks for term in set(_tokens(chunk.text)))
        hits = []
        for chunk in chunks:
            terms = Counter(_tokens(chunk.text))
            score = 0.0
            for term, qtf in query_terms.items():
                tf = terms[term]
                if tf:
                    idf = math.log(
                        1 + (len(chunks) - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5)
                    )
                    score += qtf * idf * (tf * 2.2) / (tf + 1.2)
            hits.append(SearchHit(chunk=chunk, score=score, sparse_score=score))
        return sorted(hits, key=lambda hit: hit.score, reverse=True)[:limit]


class PostgresChunkStore:
    def __init__(self, database_url: str):
        try:
            from psycopg_pool import ConnectionPool
        except ImportError as exc:
            raise RuntimeError("Install ragforge[postgres] for the PostgreSQL backend") from exc
        self.pool = ConnectionPool(database_url, min_size=1, max_size=10, open=True)

    def upsert_document(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        with self.pool.connection() as conn, conn.transaction():
            conn.execute("DELETE FROM chunks WHERE document_id = %s", (chunks[0].document_id,))
            with conn.cursor() as cur:
                cur.executemany(
                    """INSERT INTO chunks
                    (id, document_id, chunk_index, content, content_hash, source,
                     metadata, embedding)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
                    [(c.id, c.document_id, c.chunk_index, c.text, c.content_hash, c.source,
                      __import__("json").dumps(c.metadata), c.embedding) for c in chunks],
                )
        return len(chunks)

    def document_hashes(self, document_id: str) -> set[str]:
        with self.pool.connection() as conn:
            rows = conn.execute(
                "SELECT content_hash FROM chunks WHERE document_id=%s", (document_id,)
            )
            return {row[0] for row in rows}

    def all_chunks(self) -> list[Chunk]:
        return []

    @staticmethod
    def _chunk(row: tuple) -> Chunk:
        return Chunk(id=row[0], document_id=row[1], chunk_index=row[2], text=row[3],
                     content_hash=row[4], source=row[5], metadata=row[6], embedding=[])

    def dense_search(self, embedding: list[float], limit: int) -> list[SearchHit]:
        sql = """SELECT id, document_id, chunk_index, content, content_hash, source, metadata,
                 1 - (embedding <=> %s::vector) AS score FROM chunks
                 ORDER BY embedding <=> %s::vector LIMIT %s"""
        with self.pool.connection() as conn:
            rows = conn.execute(sql, (embedding, embedding, limit)).fetchall()
        return [SearchHit(self._chunk(row), row[7], dense_score=row[7]) for row in rows]

    def sparse_search(self, query: str, limit: int) -> list[SearchHit]:
        sql = """SELECT id, document_id, chunk_index, content, content_hash, source, metadata,
                 ts_rank_cd(search_vector, websearch_to_tsquery('english', %s)) AS score
                 FROM chunks WHERE search_vector @@ websearch_to_tsquery('english', %s)
                 ORDER BY score DESC LIMIT %s"""
        with self.pool.connection() as conn:
            rows = conn.execute(sql, (query, query, limit)).fetchall()
        return [SearchHit(self._chunk(row), row[7], sparse_score=row[7]) for row in rows]
