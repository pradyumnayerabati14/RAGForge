import uuid
from dataclasses import dataclass

from ragforge.chunking import chunk_text
from ragforge.embeddings import Embedder
from ragforge.models import Chunk, IngestDocument
from ragforge.store import ChunkStore


@dataclass
class IngestionStats:
    documents_seen: int = 0
    documents_updated: int = 0
    chunks_upserted: int = 0
    documents_unchanged: int = 0


class IngestionService:
    def __init__(self, store: ChunkStore, embedder: Embedder):
        self.store, self.embedder = store, embedder

    def ingest(self, documents: list[IngestDocument]) -> IngestionStats:
        stats = IngestionStats(documents_seen=len(documents))
        for document in documents:
            pieces = chunk_text(document.text)
            hashes = {piece.content_hash for piece in pieces}
            if hashes and hashes == self.store.document_hashes(document.id):
                stats.documents_unchanged += 1
                continue
            embeddings = self.embedder.embed([piece.text for piece in pieces])
            chunks = [Chunk(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{document.id}:{piece.content_hash}")),
                document_id=document.id, text=piece.text, embedding=embedding,
                chunk_index=piece.index, source=document.source, content_hash=piece.content_hash,
                metadata=document.metadata,
            ) for piece, embedding in zip(pieces, embeddings, strict=True)]
            stats.chunks_upserted += self.store.upsert_document(chunks)
            stats.documents_updated += 1
        return stats

