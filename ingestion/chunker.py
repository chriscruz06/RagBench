"""
Chunking strategies — this module is central to Phase 3 ablation experiments.

Each strategy takes a Document and returns a list of smaller Documents (chunks)
with metadata preserved + chunk-level metadata added.

Interview talking point: "I implemented three chunking strategies and benchmarked
them against each other using my eval framework. Fixed-size was the baseline,
sentence-level preserved semantic boundaries, and semantic chunking used embedding
similarity to find natural breakpoints."
"""

from dataclasses import dataclass
from ingestion.loader import Document
from config.settings import settings


def chunk_fixed(doc: Document, chunk_size: int = None, overlap: int = None) -> list[Document]:
    """
    Fixed-size character chunking with overlap.
    Simple, fast, but can split mid-sentence.
    """
    chunk_size = chunk_size or settings.chunk_size
    overlap = overlap or settings.chunk_overlap
    text = doc.text
    chunks = []

    start = 0
    chunk_idx = 0
    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]

        if chunk_text.strip():
            chunks.append(Document(
                text=chunk_text,
                metadata={
                    **doc.metadata,
                    "chunk_index": chunk_idx,
                    "chunk_strategy": "fixed",
                    "char_start": start,
                    "char_end": min(end, len(text)),
                }
            ))
            chunk_idx += 1

        start += chunk_size - overlap

    return chunks


def chunk_sentence(doc: Document) -> list[Document]:
    """
    Sentence-level chunking — split on sentence boundaries, then group
    into chunks that don't exceed chunk_size. Preserves semantic boundaries
    better than fixed chunking.
    """
    import re

    # Simple sentence splitter (handles Mr./Dr./etc. edge cases reasonably)
    sentences = re.split(r'(?<=[.!?])\s+', doc.text)

    chunks = []
    current_chunk = []
    current_len = 0
    chunk_idx = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if current_len + len(sentence) > settings.chunk_size and current_chunk:
            chunks.append(Document(
                text=" ".join(current_chunk),
                metadata={
                    **doc.metadata,
                    "chunk_index": chunk_idx,
                    "chunk_strategy": "sentence",
                }
            ))
            chunk_idx += 1

            # Keep last sentence as overlap for context continuity
            current_chunk = [current_chunk[-1]] if current_chunk else []
            current_len = len(current_chunk[0]) if current_chunk else 0

        current_chunk.append(sentence)
        current_len += len(sentence)

    # Don't forget the last chunk
    if current_chunk:
        chunks.append(Document(
            text=" ".join(current_chunk),
            metadata={
                **doc.metadata,
                "chunk_index": chunk_idx,
                "chunk_strategy": "sentence",
            }
        ))

    return chunks


def chunk_semantic(doc: Document) -> list[Document]:
    """
    Semantic chunking — use embedding similarity between consecutive
    sentences to find natural breakpoints.

    This is the most sophisticated strategy and the slowest. It embeds
    each sentence, then splits where cosine similarity between adjacent
    sentences drops below a threshold.

    Placeholder for Phase 3 — requires sentence-transformers at chunk time.
    """
    # TODO: Implement in Phase 3 ablation experiments
    # 1. Split into sentences
    # 2. Embed each sentence
    # 3. Compute cosine similarity between adjacent embeddings
    # 4. Split where similarity drops below threshold
    # 5. Group resulting segments into chunks
    raise NotImplementedError(
        "Semantic chunking is a Phase 3 feature. Use 'fixed' or 'sentence' for now."
    )


# ── Strategy dispatcher ───────────────────────────────────────────

STRATEGIES = {
    "fixed": chunk_fixed,
    "sentence": chunk_sentence,
    "semantic": chunk_semantic,
}


def chunk_documents(
    docs: list[Document],
    strategy: str = None,
) -> list[Document]:
    """
    Chunk a list of Documents using the specified strategy.

    Usage:
        chunks = chunk_documents(docs, strategy="sentence")
    """
    strategy = strategy or settings.chunk_strategy
    chunker = STRATEGIES.get(strategy)
    if not chunker:
        raise ValueError(f"Unknown strategy '{strategy}'. Options: {list(STRATEGIES.keys())}")

    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunker(doc))

    print(f"[chunking] {len(docs)} docs → {len(all_chunks)} chunks (strategy={strategy})")
    return all_chunks
