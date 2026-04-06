"""
FastAPI backend — serves the RAG pipeline over HTTP.
"""

import math
import re
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


app = FastAPI(title="RagBench", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class SourceResponse(BaseModel):
    """A single retrieved source — text plus structured metadata for the UI."""
    title: str           # e.g. "Catechism of the Catholic Church"
    section: str         # e.g. "§1213" or "Genesis 1:1"
    text: str            # the chunk text itself
    score: float         # similarity / rerank score
    doc_type: str        # "catechism" | "bible" | etc.


class DebugMetrics(BaseModel):
    """Intrinsic evaluation metrics — no ground truth required."""
    relevance: float            # mean similarity score across retrieved chunks
    faithfulness: float         # token overlap between answer and context
    context_utilization: float  # fraction of chunks cited in the answer


class QueryResponse(BaseModel):
    question: str
    answer: str
    chunks_used: int
    abstained: bool
    sources: list[SourceResponse]
    debug_metrics: DebugMetrics


# ── Mapping helpers ───────────────────────────────────────────────

DOC_TYPE_TITLES = {
    "catechism": "Catechism of the Catholic Church",
    "bible": "Sacred Scripture (Douay-Rheims)",
}


def chunk_to_source(chunk) -> SourceResponse:
    """Convert a RetrievedChunk into the API's SourceResponse shape."""
    meta = chunk.metadata or {}
    doc_type = meta.get("doc_type", "unknown")

    # Title: human-readable name of the work
    title = DOC_TYPE_TITLES.get(doc_type, meta.get("source", "Unknown"))

    # Section: the citation locator (CCC §1234, or filename for bible chunks)
    if "reference" in meta and meta["reference"]:
        section = meta["reference"]
    elif "paragraph_number" in meta and meta["paragraph_number"]:
        section = f"§{meta['paragraph_number']}"
    else:
        # Fall back to source filename for bible chunks
        section = meta.get("source", "").replace(".txt", "").replace("_", " ").title()

    return SourceResponse(
        title=title,
        section=section,
        text=chunk.text,
        score=round(float(chunk.score), 4),
        doc_type=doc_type,
    )


# ── Debug metrics — intrinsic, no ground truth needed ────────────

_TOKEN_RE = re.compile(r"\b\w+\b")
# Common stopwords excluded so faithfulness isn't dominated by "the", "of", etc.
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "of", "to",
    "in", "on", "at", "by", "for", "with", "from", "as", "that", "this",
    "these", "those", "it", "its", "we", "us", "our", "i", "you", "he",
    "she", "they", "them", "his", "her", "their", "which", "who", "whom",
    "what", "when", "where", "how", "not", "no", "so", "if", "then", "than",
    "also", "such",
}


def _sigmoid(x: float) -> float:
    """Squash an unbounded reranker score into (0, 1)."""
    # Clamp to avoid overflow on extreme scores
    x = max(-30.0, min(30.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def _tokens(text: str) -> set[str]:
    """Lowercase content tokens with stopwords removed."""
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS}


def compute_debug_metrics(answer: str, chunks: list) -> DebugMetrics:
    """
    Compute lightweight intrinsic metrics that don't need a test set.

    These are proxies — true faithfulness would require an LLM-as-judge,
    and true relevance would require human-annotated relevance grades.
    But for live debugging in the UI they give a useful signal.

    Note on relevance: chunk.score comes from the cross-encoder reranker
    and is unbounded (typically -10 to +10). We sigmoid-normalize it
    into (0, 1) so the metric stays interpretable in the UI.
    """
    if not chunks:
        return DebugMetrics(relevance=0.0, faithfulness=0.0, context_utilization=0.0)

    # ── Relevance: mean of sigmoid-normalized rerank scores ──
    normalized_scores = [_sigmoid(c.score) for c in chunks]
    relevance = sum(normalized_scores) / len(normalized_scores)

    # ── Faithfulness: content-token overlap (Jaccard-style) ──
    answer_tokens = _tokens(answer)
    context_tokens = set()
    for c in chunks:
        context_tokens |= _tokens(c.text)

    if not answer_tokens:
        faithfulness = 0.0
    else:
        overlap = answer_tokens & context_tokens
        faithfulness = len(overlap) / len(answer_tokens)

    # ── Context utilization: fraction of chunks cited in the answer ──
    cited = 0
    for c in chunks:
        meta = c.metadata or {}
        para_num = meta.get("paragraph_number")
        ref = meta.get("reference", "")
        # Look for either the paragraph number or the bare reference
        if para_num and (
            f"§{para_num}" in answer or f"CCC {para_num}" in answer
        ):
            cited += 1
        elif ref and ref in answer:
            cited += 1

    context_utilization = cited / len(chunks) if chunks else 0.0

    return DebugMetrics(
        relevance=round(float(relevance), 4),
        faithfulness=round(float(faithfulness), 4),
        context_utilization=round(float(context_utilization), 4),
    )


# ── Routes ────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def handle_query(req: QueryRequest):
    from pipeline import query
    result = query(req.question, top_k=req.top_k)

    chunks = result.get("retrieved_chunks", [])
    sources = [chunk_to_source(c) for c in chunks]
    debug_metrics = compute_debug_metrics(result["answer"], chunks)

    return QueryResponse(
        question=result["question"],
        answer=result["answer"],
        chunks_used=result["chunks_used"],
        abstained=result["abstained"],
        sources=sources,
        debug_metrics=debug_metrics,
    )


@app.post("/ingest")
def handle_ingest(doc_type: str = "unknown"):
    """Trigger ingestion (for dev use — production would handle file uploads)."""
    from pipeline import ingest
    result = ingest(f"data/processed/{doc_type}", doc_type=doc_type)
    return result


# Run: uvicorn api.app:app --reload --port 8000
