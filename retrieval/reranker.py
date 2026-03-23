"""
Cross-encoder reranker — the single biggest quality lever in a RAG pipeline.

Why this matters:
- Bi-encoder embeddings (what ChromaDB uses) are fast but approximate. They
  compress query and document into independent vectors, losing fine-grained
  interaction between them.
- A cross-encoder takes (query, document) as a PAIR, letting the model attend
  across both simultaneously. This is much more accurate for relevance scoring
  but too slow to run over the entire corpus.

The pattern: retrieve a wide pool with the bi-encoder (fast, ~top 20),
then rerank that pool with the cross-encoder (accurate, ~top 5).

This is standard in production RAG systems (Cohere Rerank, Jina Reranker, etc.)
but we're using an open-source cross-encoder so it runs locally.
"""

from dataclasses import dataclass
from config.settings import settings


# Lazy-loaded singleton so we don't reload the model on every query
_reranker_model = None


def get_reranker():
    """Load the cross-encoder model (cached after first call)."""
    global _reranker_model
    if _reranker_model is None:
        from sentence_transformers import CrossEncoder

        _reranker_model = CrossEncoder(
            settings.reranker_model,
            max_length=512,
        )
        print(f"[reranker] Loaded cross-encoder: {settings.reranker_model}")
    return _reranker_model


def rerank(
    query: str,
    documents: list[dict],
    top_k: int = 5,
) -> list[dict]:
    """
    Rerank a list of candidate documents using the cross-encoder.

    Args:
        query: The user's question
        documents: List of dicts with at least 'text' key
                   (typically RetrievedChunk-like dicts)
        top_k: Number of top results to return after reranking

    Returns:
        The top_k documents, re-sorted by cross-encoder relevance score.
        Each dict gets a 'rerank_score' field added.
    """
    if not documents:
        return []

    reranker = get_reranker()

    # Build (query, document) pairs for the cross-encoder
    pairs = [(query, doc["text"]) for doc in documents]

    # Score all pairs — this is the expensive part, but we're only
    # scoring ~20 candidates, not the whole corpus
    scores = reranker.predict(pairs)

    # Attach scores and sort
    for doc, score in zip(documents, scores):
        doc["rerank_score"] = float(score)

    ranked = sorted(documents, key=lambda d: d["rerank_score"], reverse=True)

    return ranked[:top_k]
