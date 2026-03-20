"""
FastAPI backend — serves the RAG pipeline over HTTP.
"""

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


app = FastAPI(title="RagBench", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class QueryResponse(BaseModel):
    question: str
    answer: str
    chunks_used: int
    abstained: bool
    sources: list[dict]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def handle_query(req: QueryRequest):
    from pipeline import query
    result = query(req.question, top_k=req.top_k)
    return QueryResponse(
        question=result["question"],
        answer=result["answer"],
        chunks_used=result["chunks_used"],
        abstained=result["abstained"],
        sources=result["sources"],
    )


@app.post("/ingest")
def handle_ingest(doc_type: str = "unknown"):
    """Trigger ingestion (for dev use — production would handle file uploads)."""
    from pipeline import ingest
    result = ingest(f"data/raw/{doc_type}", doc_type=doc_type)
    return result


# Run: uvicorn api.app:app --reload --port 8000
