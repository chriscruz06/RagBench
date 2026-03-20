"""
Retrieval — query the vector store and return ranked chunks.

The similarity_threshold is a key design decision: if no chunk scores
above it, the pipeline abstains rather than hallucinating. This is
the "Layer 4" defense against unfaithful generation.
"""

from dataclasses import dataclass, field
from config.settings import settings
from ingestion.embedder import get_embedding_function, get_collection


@dataclass
class RetrievedChunk:
    """A chunk returned by retrieval, with its similarity score."""
    text: str
    score: float
    metadata: dict = field(default_factory=dict)


def retrieve(
    query: str,
    top_k: int = None,
    threshold: float = None,
    collection_name: str = "ragbench",
) -> list[RetrievedChunk]:
    """
    Embed the query and retrieve the top-K most similar chunks.

    Chunks below the similarity threshold are filtered out.
    If no chunks survive filtering, the generation layer should abstain.

    Returns:
        List of RetrievedChunk, sorted by descending similarity score.
    """
    top_k = top_k or settings.top_k
    threshold = threshold or settings.similarity_threshold

    embedding_fn = get_embedding_function()
    query_embedding = embedding_fn.embed_query(query)

    collection = get_collection(collection_name)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        # ChromaDB cosine distance: 0 = identical, 2 = opposite
        # Convert to similarity: 1 - (distance / 2)
        similarity = 1 - (distance / 2)

        if similarity >= threshold:
            chunks.append(RetrievedChunk(
                text=doc,
                score=similarity,
                metadata=meta,
            ))

    chunks.sort(key=lambda c: c.score, reverse=True)

    print(f"[retrieval] Query returned {len(chunks)} chunks above threshold "
          f"(top_k={top_k}, threshold={threshold})")

    return chunks
