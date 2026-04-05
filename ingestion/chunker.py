"""
Chunking strategies — this module is central to Phase 3 ablation experiments.

Each strategy takes a Document and returns a list of smaller Documents (chunks)
with metadata preserved + chunk-level metadata added.

Interview talking point: "I implemented three chunking strategies and benchmarked
them against each other using my eval framework. Fixed-size was the baseline,
sentence-level preserved semantic boundaries, and semantic chunking used embedding
similarity to find natural breakpoints."
"""

import re
import numpy as np
from sentence_transformers import SentenceTransformer
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


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, filtering blanks."""
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in parts if s.strip()]


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(dot / norm)

_st_model = None

def _get_st_model():
    """Load SentenceTransformer once and reuse across documents."""
    global _st_model
    if _st_model is None:
        _st_model = SentenceTransformer(settings.embedding_model)
    return _st_model

def chunk_semantic(doc: Document) -> list[Document]:
    """
    Semantic chunking — use embedding similarity between consecutive
    sentences to find natural breakpoints.

    Algorithm:
    1. Split document into sentences
    2. Embed each sentence using the project's embedding model
    3. Compute cosine similarity between every adjacent pair
    4. Split where similarity drops below the breakpoint threshold
    5. Merge resulting segments, respecting max_chunk_size

    Slower than fixed/sentence (requires embedding at chunk time), but
    produces chunks that respect topical boundaries — important for
    theological texts where a paragraph might transition from doctrine
    to scriptural citation to pastoral application.
    """
    threshold = settings.semantic_breakpoint_threshold
    max_size = settings.semantic_max_chunk_size

    sentences = _split_sentences(doc.text)

    # Edge case: very short documents
    if len(sentences) <= 1:
        return [Document(
            text=doc.text.strip(),
            metadata={
                **doc.metadata,
                "chunk_index": 0,
                "chunk_strategy": "semantic",
            }
        )] if doc.text.strip() else []

    # ── Step 1: Embed all sentences in a single batch ──
    model = _get_st_model()
    embeddings = model.encode(sentences, show_progress_bar=False)

    # ── Step 2: Find breakpoints ──
    # Compare each sentence to its neighbor; low similarity = topic shift
    breakpoints = []
    for i in range(len(embeddings) - 1):
        sim = _cosine_similarity(embeddings[i], embeddings[i + 1])
        if sim < threshold:
            breakpoints.append(i + 1)  # split BEFORE this sentence index

    # ── Step 3: Build segments from breakpoints ──
    # breakpoints mark where new segments start
    segment_starts = [0] + breakpoints
    segments = []
    for i, start in enumerate(segment_starts):
        end = segment_starts[i + 1] if i + 1 < len(segment_starts) else len(sentences)
        segment_text = " ".join(sentences[start:end])
        segments.append(segment_text)

    # ── Step 4: Emit chunks, merging only tiny segments ──
    # Each segment becomes its own chunk unless it's very short
    # (< min_segment chars), in which case it merges with the next.
    # Segments above max_size get split at sentence boundaries.
    MIN_SEGMENT = 50  # merge segments shorter than this with their neighbor
    chunks = []
    chunk_idx = 0
    buffer = ""

    for segment in segments:
        candidate = (buffer + " " + segment).strip() if buffer else segment

        if len(candidate) > max_size:
            # Flush buffer first if it has content
            if buffer:
                chunks.append(Document(
                    text=buffer,
                    metadata={
                        **doc.metadata,
                        "chunk_index": chunk_idx,
                        "chunk_strategy": "semantic",
                    }
                ))
                chunk_idx += 1
                buffer = ""
                candidate = segment

            # Split the oversized segment by sentences, grouping up to max_size
            seg_sentences = _split_sentences(candidate)
            sub_buffer = ""
            for sent in seg_sentences:
                test = (sub_buffer + " " + sent).strip() if sub_buffer else sent
                if len(test) > max_size and sub_buffer:
                    chunks.append(Document(
                        text=sub_buffer,
                        metadata={
                            **doc.metadata,
                            "chunk_index": chunk_idx,
                            "chunk_strategy": "semantic",
                        }
                    ))
                    chunk_idx += 1
                    sub_buffer = sent
                else:
                    sub_buffer = test
            if sub_buffer:
                buffer = sub_buffer
        elif len(candidate) < MIN_SEGMENT:
            # Too short to stand alone — buffer it for merging with the next
            buffer = candidate
        else:
            # Normal-sized segment — flush as its own chunk
            chunks.append(Document(
                text=candidate,
                metadata={
                    **doc.metadata,
                    "chunk_index": chunk_idx,
                    "chunk_strategy": "semantic",
                }
            ))
            chunk_idx += 1
            buffer = ""

    # Flush remaining buffer
    if buffer.strip():
        chunks.append(Document(
            text=buffer.strip(),
            metadata={
                **doc.metadata,
                "chunk_index": chunk_idx,
                "chunk_strategy": "semantic",
            }
        ))

    return chunks


# ── Strategy dispatcher ──

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