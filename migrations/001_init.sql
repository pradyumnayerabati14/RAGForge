CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id uuid PRIMARY KEY,
    document_id text NOT NULL,
    chunk_index integer NOT NULL,
    content text NOT NULL,
    content_hash char(64) NOT NULL,
    source text NOT NULL,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    embedding vector(384) NOT NULL,
    search_vector tsvector GENERATED ALWAYS AS
        (to_tsvector('english', coalesce(content, ''))) STORED,
    updated_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (document_id, content_hash)
);

CREATE INDEX IF NOT EXISTS chunks_document_id_idx ON chunks (document_id);
CREATE INDEX IF NOT EXISTS chunks_search_idx ON chunks USING gin (search_vector);
CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks
    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);

