from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from ragforge.config import get_settings
from ragforge.container import get_container
from ragforge.models import AskRequest, AskResponse, IngestRequest, IngestResponse


@asynccontextmanager
async def lifespan(_: FastAPI):
    get_container()
    yield


app = FastAPI(title="RAGForge", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "backend": get_settings().backend}


@app.post("/v1/documents", response_model=IngestResponse)
def ingest(request: IngestRequest) -> IngestResponse:
    stats = get_container().ingestion.ingest(request.documents)
    return IngestResponse(**vars(stats))


@app.post("/v1/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    try:
        return get_container().qa.ask(request.question, request.top_k or get_settings().top_k)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Question answering failed") from exc

