"""
Retrieval | two-stage pipeline: wide vector search → cross-encoder rerank.

Stage 1 (bi-encoder): Fast approximate search over the full corpus via ChromaDB.
  Retrieves a wide candidate pool (retrieval_top_k, default 20).

Stage 2 (cross-encoder reranker): Scores each (query, chunk) pair with a
  cross-encoder for much more accurate relevance. Returns the final top_k.

The similarity_threshold is applied AFTER reranking, using the reranker's
score if reranking is enabled, or the embedding similarity if not.
This is the "Layer 4" defense against unfaithful generation.
"""

from dataclasses import dataclass, field
from config.settings import settings
from ingestion.embedder import get_embedding_function, get_collection


@dataclass
class RetrievedChunk:
    """A chunk returned by retrieval, with its similarity score."""
    text: str
    score: float  # final score (reranker score if reranked, else embedding similarity)
    metadata: dict = field(default_factory=dict)
    embedding_score: float = 0.0  # original bi-encoder similarity
    rerank_score: float | None = None  # cross-encoder score (None if reranking disabled)


def retrieve(
    query: str,
    top_k: int = None,
    threshold: float = None,
    collection_name: str = "ragbench",
) -> list[RetrievedChunk]:
    """
    Embed the query and retrieve the most relevant chunks.

    Two-stage pipeline:
    1. Retrieve a wide candidate pool from ChromaDB (retrieval_top_k)
    2. Rerank candidates with cross-encoder (if enabled)
    3. Apply similarity threshold and return final top_k

    Chunks below the similarity threshold are filtered out.
    If no chunks survive filtering, the generation layer should abstain.

    Returns:
        List of RetrievedChunk, sorted by descending relevance score.
    """
    top_k = top_k or settings.top_k
    threshold = threshold or settings.similarity_threshold

    embedding_fn = get_embedding_function()
    query_embedding = embedding_fn.embed_query(query)

    collection = get_collection(collection_name)

    # ── Stage 1: Wide retrieval from vector store ────────────
    # Retrieve more candidates than we need so the reranker has
    # a good pool to pick from
    retrieval_k = settings.retrieval_top_k if settings.use_reranker else top_k

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=retrieval_k,
        include=["documents", "metadatas", "distances"],
    )

    candidates = []
    for doc, meta, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        # ChromaDB cosine distance: 0 = identical, 2 = opposite
        # Convert to similarity: 1 - (distance / 2)
        similarity = 1 - (distance / 2)

        candidates.append({
            "text": doc,
            "metadata": meta,
            "embedding_score": similarity,
        })

    print(f"[retrieval] Stage 1: {len(candidates)} candidates from vector search")

    # ── Stage 2: Rerank with cross-encoder ───────────────────
    if settings.use_reranker and candidates:
        from retrieval.reranker import rerank

        reranked = rerank(query, candidates, top_k=top_k)

        chunks = []
        for item in reranked:
            # Use reranker score as the primary score
            # Cross-encoder scores are typically in [-10, 10] range,
            # we don't apply the same threshold as embedding similarity
            chunks.append(RetrievedChunk(
                text=item["text"],
                score=item["rerank_score"],
                metadata=item["metadata"],
                embedding_score=item["embedding_score"],
                rerank_score=item["rerank_score"],
            ))

        # Filter by reranker threshold (different scale than embedding similarity)
        chunks = [c for c in chunks if c.rerank_score >= settings.reranker_threshold]

        print(f"[retrieval] Stage 2: {len(chunks)} chunks after reranking "
              f"(threshold={settings.reranker_threshold})")
    else:
        # No reranker, use embedding similarity with threshold
        chunks = []
        for item in candidates:
            if item["embedding_score"] >= threshold:
                chunks.append(RetrievedChunk(
                    text=item["text"],
                    score=item["embedding_score"],
                    metadata=item["metadata"],
                    embedding_score=item["embedding_score"],
                ))

        chunks.sort(key=lambda c: c.score, reverse=True)
        chunks = chunks[:top_k]

        print(f"[retrieval] {len(chunks)} chunks above threshold "
              f"(top_k={top_k}, threshold={threshold})")

    return chunks