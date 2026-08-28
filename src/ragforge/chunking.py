import hashlib
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class TextChunk:
    index: int
    text: str
    content_hash: str


def content_hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode()).hexdigest()


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 120) -> list[TextChunk]:
    if chunk_size <= overlap or overlap < 0:
        raise ValueError("chunk_size must be greater than overlap")
    clean = re.sub(r"\s+", " ", text).strip()
    if not clean:
        return []
    chunks: list[TextChunk] = []
    start = 0
    while start < len(clean):
        end = min(start + chunk_size, len(clean))
        if end < len(clean):
            boundary = clean.rfind(" ", start + chunk_size // 2, end)
            if boundary > start:
                end = boundary
        piece = clean[start:end].strip()
        chunks.append(TextChunk(len(chunks), piece, content_hash(piece)))
        if end == len(clean):
            break
        start = end - overlap
    return chunks

