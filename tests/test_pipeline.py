from ragforge.embeddings import HashingEmbedder
from ragforge.generation import ExtractiveGenerator
from ragforge.ingestion import IngestionService
from ragforge.models import IngestDocument
from ragforge.retrieval import HybridRetriever, LexicalReranker
from ragforge.service import QAService
from ragforge.store import InMemoryChunkStore


def pipeline():
    store = InMemoryChunkStore()
    embedder = HashingEmbedder(128)
    ingestion = IngestionService(store, embedder)
    qa = QAService(HybridRetriever(store, embedder, LexicalReranker()), ExtractiveGenerator())
    return store, ingestion, qa


def test_ingestion_is_incremental_and_answers_with_citations():
    store, ingestion, qa = pipeline()
    docs = [IngestDocument(id="mars", source="space.md",
             text="Mars is known as the red planet because iron minerals oxidize in its soil.")]
    initial = ingestion.ingest(docs)
    repeated = ingestion.ingest(docs)
    assert initial.documents_updated == 1
    assert repeated.documents_unchanged == 1
    assert len(store.all_chunks()) == 1
    answer = qa.ask("Why is Mars called the red planet?")
    assert answer.citations[0].source == "space.md"
    assert "[1]" in answer.answer
    assert answer.latency_ms >= 0


def test_changed_document_replaces_old_chunks():
    store, ingestion, _ = pipeline()
    ingestion.ingest([IngestDocument(id="doc", text="old content")])
    ingestion.ingest([IngestDocument(id="doc", text="new content")])
    assert len(store.all_chunks()) == 1
    assert store.all_chunks()[0].text == "new content"

