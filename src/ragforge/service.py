import time

from ragforge.generation import Generator
from ragforge.models import AskResponse, Citation
from ragforge.retrieval import HybridRetriever


class QAService:
    def __init__(self, retriever: HybridRetriever, generator: Generator):
        self.retriever, self.generator = retriever, generator

    def ask(self, question: str, top_k: int = 5) -> AskResponse:
        started = time.perf_counter()
        hits = self.retriever.search(question, top_k)
        retrieved = time.perf_counter()
        answer = self.generator.generate(question, hits)
        generated = time.perf_counter()
        citations = [Citation(id=i, document_id=h.chunk.document_id, source=h.chunk.source,
                              chunk_index=h.chunk.chunk_index, excerpt=h.chunk.text[:280],
                              score=round(h.rerank_score or h.score, 6))
                     for i, h in enumerate(hits, 1)]
        return AskResponse(answer=answer, citations=citations,
                           latency_ms=round((generated-started)*1000, 2),
                           retrieval_ms=round((retrieved-started)*1000, 2),
                           generation_ms=round((generated-retrieved)*1000, 2))

