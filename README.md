# RAGForge

A production-oriented document QA service built around incremental indexing, hybrid
retrieval, cross-encoder reranking, citation-grounded generation, and measurable
answer quality. It includes a FastAPI serving layer, a PySpark batch ingestion job,
pgvector persistence, and a RAGAS evaluation harness.

## Architecture

```text
Documents / object storage
          |
     PySpark normalize + partition
          |
  incremental content hashing
          |
    chunk + embed only changes
          v
PostgreSQL + pgvector + full-text index
          |
 BM25-style FTS ---- dense HNSW search
          \          /
       reciprocal rank fusion
                |
       cross-encoder reranker
                |
 citation-constrained LLM generation
                |
 answer + sources + latency telemetry
```

The default `memory` backend uses a deterministic feature-hashing embedder and an
extractive generator, so the complete flow runs without cloud keys. The `postgres`
backend enables pgvector, transformer embeddings and cross-encoder reranking. Add an
OpenAI key to use LangChain-powered generation; without one, answers remain extractive.

## Quick start (local, no infrastructure)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
uvicorn ragforge.api:app --reload
```

In another terminal:

```bash
curl -X POST http://localhost:8000/v1/documents \
  -H 'content-type: application/json' \
  -d '{"documents":[{"id":"mars","source":"space.md","text":"Mars is red because iron minerals in its soil oxidize."}]}'

curl -X POST http://localhost:8000/v1/ask \
  -H 'content-type: application/json' \
  -d '{"question":"Why is Mars red?"}'
```

Interactive API documentation is available at `http://localhost:8000/docs`.

## Full stack

```bash
cp .env.example .env
# Optionally set RAGFORGE_OPENAI_API_KEY in .env
docker compose up --build
```

The PostgreSQL migration installs pgvector, a 384-dimensional HNSW vector index,
and a generated English full-text `tsvector` GIN index. Changed documents atomically
replace their prior chunks; documents whose chunk hashes are unchanged are skipped.

## Distributed ingestion

Input may be a text/Markdown glob, local path, S3 path (with the matching Hadoop
connector), or JSONL containing `id`, `text`, and `source` fields.

```bash
pip install -e ".[spark]"
spark-submit spark/ingest.py --input 'data/**/*.md' --api-url http://localhost:8000
```

## Evaluation

Create a JSONL file with `question` and `ground_truth` fields, start the API, and run:

```bash
pip install -e ".[eval]"
python eval/run_ragas.py data/eval.jsonl --output reports/ragas.json
```

The API reports retrieval, generation, and end-to-end latency on every response.
The harness measures faithfulness and answer relevancy. The résumé figures (0.72 to
0.89 faithfulness) should be recorded only after reproducing them on your own dataset;
this repository provides the harness rather than fabricating benchmark output.

## API

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Readiness and backend status |
| `POST` | `/v1/documents` | Incrementally index document batches |
| `POST` | `/v1/ask` | Hybrid retrieve, rerank, and answer with citations |

## Design notes

- Reciprocal rank fusion combines dense and sparse ranks without assuming comparable
  raw scores.
- Stable UUIDs derive from document IDs and content hashes, making retries idempotent.
- Retrieved text is treated as untrusted context, and the generation prompt explicitly
  rejects instructions embedded in documents.
- The online endpoint never exposes internal exception details.
- The in-memory backend is intentionally process-local; use PostgreSQL in deployment.

## Development

```bash
make install
make lint
make test
```

Licensed under the MIT License.

