from functools import lru_cache

from ragforge.config import get_settings
from ragforge.embeddings import HashingEmbedder, SentenceTransformerEmbedder
from ragforge.generation import ExtractiveGenerator, OpenAIGenerator
from ragforge.ingestion import IngestionService
from ragforge.retrieval import CrossEncoderReranker, HybridRetriever, LexicalReranker
from ragforge.service import QAService
from ragforge.store import InMemoryChunkStore, PostgresChunkStore


class Container:
    def __init__(self):
        settings = get_settings()
        if settings.backend == "postgres":
            self.store = PostgresChunkStore(settings.database_url)
            self.embedder = SentenceTransformerEmbedder(settings.embedding_model)
            reranker = CrossEncoderReranker(settings.reranker_model)
        else:
            self.store = InMemoryChunkStore()
            self.embedder = HashingEmbedder(settings.embedding_dimensions)
            reranker = LexicalReranker()
        generator = (OpenAIGenerator(settings.openai_api_key, settings.openai_model)
                     if settings.openai_api_key else ExtractiveGenerator())
        self.ingestion = IngestionService(self.store, self.embedder)
        self.qa = QAService(HybridRetriever(self.store, self.embedder, reranker,
                                            settings.candidate_k), generator)


@lru_cache
def get_container() -> Container:
    return Container()

