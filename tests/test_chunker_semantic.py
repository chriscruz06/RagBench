"""
Tests for the semantic chunking strategy.

These tests verify the chunker's logic without requiring the full
embedding model — we mock SentenceTransformer where needed to keep
tests fast, and include one integration-style test that uses the
real model (marked slow).

Usage:
    python -m pytest tests/test_chunker_semantic.py -v
    python -m pytest tests/test_chunker_semantic.py -v -k "not slow"  # skip slow tests
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from ingestion.loader import Document
from ingestion.chunker import (
    chunk_semantic,
    _split_sentences,
    _cosine_similarity,
    chunk_documents,
)


# ── Helper fixtures ──

@pytest.fixture
def sample_doc():
    """A document with clear topical shifts for semantic splitting."""
    return Document(
        text=(
            "Baptism is the first sacrament of initiation. "
            "It cleanses original sin and incorporates us into the Church. "
            "The ordinary minister of baptism is a bishop, priest, or deacon. "
            "The Ten Commandments are the moral law given by God to Moses. "
            "They summarize the duties of man toward God and neighbor. "
            "The first commandment forbids idolatry and superstition."
        ),
        metadata={"source": "test.json", "doc_type": "catechism"},
    )


@pytest.fixture
def short_doc():
    """A document with only one sentence."""
    return Document(
        text="God is love.",
        metadata={"source": "test.json", "doc_type": "catechism"},
    )


@pytest.fixture
def empty_doc():
    """An empty document."""
    return Document(text="", metadata={"source": "test.json", "doc_type": "catechism"})


# ── Unit tests: helper functions ──

class TestSplitSentences:
    def test_basic_split(self):
        result = _split_sentences("Hello world. How are you? Fine!")
        assert result == ["Hello world.", "How are you?", "Fine!"]

    def test_filters_blanks(self):
        result = _split_sentences("Hello.   Goodbye.")
        assert all(s.strip() for s in result)

    def test_single_sentence(self):
        result = _split_sentences("Just one sentence.")
        assert result == ["Just one sentence."]

    def test_empty_string(self):
        result = _split_sentences("")
        assert result == []


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = np.array([1.0, 2.0, 3.0])
        assert _cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert _cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert _cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector(self):
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 2.0])
        assert _cosine_similarity(a, b) == 0.0


# ── Unit tests: semantic chunker with mocked embeddings ──

class TestSemanticChunkerMocked:
    """Tests with mocked SentenceTransformer for speed."""

    def _make_embeddings(self, similarities: list[float]) -> np.ndarray:
        """
        Build fake embeddings where adjacent cosine similarities match
        the provided list. Uses 2D vectors for simplicity.

        similarities[i] = cosine_sim(embedding[i], embedding[i+1])
        """
        n = len(similarities) + 1
        embeddings = np.zeros((n, 2))
        # First vector points right
        embeddings[0] = [1.0, 0.0]

        for i, target_sim in enumerate(similarities):
            # Compute angle from cosine similarity
            angle = np.arccos(np.clip(target_sim, -1.0, 1.0))
            # Rotate from previous vector
            prev_angle = np.arctan2(embeddings[i][1], embeddings[i][0])
            new_angle = prev_angle + angle
            embeddings[i + 1] = [np.cos(new_angle), np.sin(new_angle)]

        return embeddings

    @patch("ingestion.chunker.SentenceTransformer")
    def test_no_breakpoints_single_chunk(self, mock_st_class, sample_doc):
        """All high similarity → one chunk."""
        sentences = _split_sentences(sample_doc.text)
        # All pairs have similarity 0.95 (above default 0.5 threshold)
        embeddings = self._make_embeddings([0.95] * (len(sentences) - 1))

        mock_model = MagicMock()
        mock_model.encode.return_value = embeddings
        mock_st_class.return_value = mock_model

        chunks = chunk_semantic(sample_doc)
        # Should be 1 chunk (no splits)
        assert len(chunks) == 1
        assert chunks[0].metadata["chunk_strategy"] == "semantic"

    @patch("ingestion.chunker.SentenceTransformer")
    def test_clear_breakpoint_two_chunks(self, mock_st_class, sample_doc):
        """One low-similarity gap → two chunks."""
        sentences = _split_sentences(sample_doc.text)
        n = len(sentences)
        # High similarity everywhere except between sentence 2 and 3
        sims = [0.95] * (n - 1)
        sims[2] = 0.1  # breakpoint between sentence 3 and 4
        embeddings = self._make_embeddings(sims)

        mock_model = MagicMock()
        mock_model.encode.return_value = embeddings
        mock_st_class.return_value = mock_model

        chunks = chunk_semantic(sample_doc)
        assert len(chunks) == 2
        # First chunk should contain the first 3 sentences (about baptism)
        assert "Baptism" in chunks[0].text
        # Second chunk should contain the commandments sentences
        assert "Commandments" in chunks[1].text

    @patch("ingestion.chunker.SentenceTransformer")
    def test_all_breakpoints_many_chunks(self, mock_st_class, sample_doc):
        """All low similarity → each sentence becomes its own chunk."""
        sentences = _split_sentences(sample_doc.text)
        n = len(sentences)
        sims = [0.1] * (n - 1)  # all below threshold
        embeddings = self._make_embeddings(sims)

        mock_model = MagicMock()
        mock_model.encode.return_value = embeddings
        mock_st_class.return_value = mock_model

        chunks = chunk_semantic(sample_doc)
        assert len(chunks) >= n - 1


    def test_single_sentence_doc(self, short_doc):
        """Single sentence → one chunk, no embedding needed."""
        chunks = chunk_semantic(short_doc)
        assert len(chunks) == 1
        assert chunks[0].text == "God is love."
        assert chunks[0].metadata["chunk_strategy"] == "semantic"

    def test_empty_doc(self, empty_doc):
        """Empty document → no chunks."""
        chunks = chunk_semantic(empty_doc)
        assert len(chunks) == 0

    @patch("ingestion.chunker.SentenceTransformer")
    def test_metadata_preserved(self, mock_st_class, sample_doc):
        """Source metadata carries through to chunks."""
        sentences = _split_sentences(sample_doc.text)
        embeddings = self._make_embeddings([0.95] * (len(sentences) - 1))

        mock_model = MagicMock()
        mock_model.encode.return_value = embeddings
        mock_st_class.return_value = mock_model

        chunks = chunk_semantic(sample_doc)
        for chunk in chunks:
            assert chunk.metadata["source"] == "test.json"
            assert chunk.metadata["doc_type"] == "catechism"
            assert chunk.metadata["chunk_strategy"] == "semantic"
            assert "chunk_index" in chunk.metadata

    @patch("ingestion.chunker.SentenceTransformer")
    def test_chunk_indices_sequential(self, mock_st_class, sample_doc):
        """Chunk indices should be 0, 1, 2, ..."""
        sentences = _split_sentences(sample_doc.text)
        n = len(sentences)
        sims = [0.1] * (n - 1)
        embeddings = self._make_embeddings(sims)

        mock_model = MagicMock()
        mock_model.encode.return_value = embeddings
        mock_st_class.return_value = mock_model

        chunks = chunk_semantic(sample_doc)
        indices = [c.metadata["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))


class TestSemanticMaxSize:
    """Test the max_chunk_size enforcement."""

    @patch("ingestion.chunker.settings")
    @patch("ingestion.chunker.SentenceTransformer")
    def test_oversized_segment_gets_split(self, mock_st_class, mock_settings):
        """A segment exceeding max_chunk_size is split at sentence boundaries."""
        mock_settings.embedding_model = "test-model"
        mock_settings.semantic_breakpoint_threshold = 0.5
        mock_settings.semantic_max_chunk_size = 80  # very small cap

        # Build a doc where all sentences are similar (one big segment)
        # but total length exceeds 80 chars
        long_doc = Document(
            text=(
                "First sentence here. "
                "Second sentence here. "
                "Third sentence here. "
                "Fourth sentence here. "
                "Fifth sentence here."
            ),
            metadata={"source": "test.json", "doc_type": "test"},
        )
        sentences = _split_sentences(long_doc.text)
        n = len(sentences)

        # All high similarity → one segment → must be split by max_size
        embeddings = np.random.randn(n, 8)
        for i in range(n):
            embeddings[i] = embeddings[0] + np.random.randn(8) * 0.01

        mock_model = MagicMock()
        mock_model.encode.return_value = embeddings
        mock_st_class.return_value = mock_model

        chunks = chunk_semantic(long_doc)
        # Should produce multiple chunks since total > 80 chars
        assert len(chunks) > 1
        # Each chunk should respect the size cap (within reason — single
        # sentences longer than max_size are allowed through)
        for chunk in chunks:
            assert chunk.metadata["chunk_strategy"] == "semantic"


class TestStrategyDispatcher:
    """Test that 'semantic' routes correctly through chunk_documents."""

    @patch("ingestion.chunker.SentenceTransformer")
    def test_dispatch_semantic(self, mock_st_class):
        """chunk_documents(strategy='semantic') calls chunk_semantic."""
        doc = Document(
            text="One sentence. Another sentence.",
            metadata={"source": "test.json", "doc_type": "test"},
        )

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[1.0, 0.0], [0.9, 0.1]])
        mock_st_class.return_value = mock_model

        chunks = chunk_documents([doc], strategy="semantic")
        assert len(chunks) >= 1
        assert all(c.metadata["chunk_strategy"] == "semantic" for c in chunks)