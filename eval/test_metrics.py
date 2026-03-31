"""
Unit tests for eval/metrics.py

Run:
    python -m pytest eval/test_metrics.py -v
    or
    python eval/test_metrics.py
"""

import sys
from pathlib import Path

# ── Tests ─────────────────────────────────────────────────────────

def test_normalize_reference():
    from eval.metrics import _normalize_reference
    assert _normalize_reference("CCC §1213") == "ccc 1213"
    assert _normalize_reference("CCC  §1213") == "ccc 1213"
    assert _normalize_reference("Romans 6:3-4") == "romans 6:3-4"
    assert _normalize_reference("  1 Cor 13:4 ") == "1 cor 13:4"
    print("  ✓ _normalize_reference")


def test_chunk_matches_source():
    from eval.metrics import _chunk_matches_source

    # CCC reference match
    chunk = {"reference": "CCC §1213", "paragraph_number": 1213}
    assert _chunk_matches_source(chunk, "CCC §1213") is True

    # Paragraph number match
    chunk = {"paragraph_number": 1213, "source": "ccc_paragraphs.json"}
    assert _chunk_matches_source(chunk, "CCC §1213") is True

    # No match
    chunk = {"reference": "CCC §999", "paragraph_number": 999}
    assert _chunk_matches_source(chunk, "CCC §1213") is False

    # Scripture source match
    chunk = {"source": "romans.txt", "doc_type": "bible"}
    assert _chunk_matches_source(chunk, "Romans 6:3-4") is True

    print("  ✓ _chunk_matches_source")


def test_precision_at_k():
    from eval.metrics import precision_at_k

    # 2 of 4 retrieved chunks are relevant
    chunks = [
        {"reference": "CCC §1213"},
        {"reference": "CCC §999"},
        {"reference": "CCC §1265"},
        {"reference": "CCC §42"},
    ]
    expected = ["CCC §1213", "CCC §1265", "Romans 6:3-4"]
    assert precision_at_k(chunks, expected) == 0.5

    # Empty retrieval
    assert precision_at_k([], expected) == 0.0

    print("  ✓ precision_at_k")


def test_recall_at_k():
    from eval.metrics import recall_at_k

    # Retrieved 2 of 3 expected sources
    chunks = [
        {"reference": "CCC §1213"},
        {"reference": "CCC §1265"},
        {"reference": "CCC §42"},
    ]
    expected = ["CCC §1213", "CCC §1265", "Romans 6:3-4"]
    result = recall_at_k(chunks, expected)
    assert abs(result - 2 / 3) < 0.001

    # Nothing expected
    assert recall_at_k(chunks, []) == 1.0

    print("  ✓ recall_at_k")


def test_mrr():
    from eval.metrics import mean_reciprocal_rank

    # First relevant chunk is at rank 1
    chunks = [
        {"reference": "CCC §1213"},
        {"reference": "CCC §999"},
    ]
    expected = ["CCC §1213"]
    assert mean_reciprocal_rank(chunks, expected) == 1.0

    # First relevant chunk is at rank 3
    chunks = [
        {"reference": "CCC §42"},
        {"reference": "CCC §99"},
        {"reference": "CCC §1213"},
    ]
    assert abs(mean_reciprocal_rank(chunks, expected) - 1 / 3) < 0.001

    # No relevant chunks
    chunks = [{"reference": "CCC §42"}]
    assert mean_reciprocal_rank(chunks, expected) == 0.0

    print("  ✓ mean_reciprocal_rank")


def test_bleu_score():
    from eval.metrics import bleu_score

    # Identical strings → high score
    text = "The Eucharist is the sacrament of the body and blood of Christ."
    score = bleu_score(text, text)
    assert score == 1.0

    # Completely different → 0
    score = bleu_score("hello world", "completely unrelated text here")
    assert score == 0.0

    # Partial overlap → between 0 and 1
    gen = "Baptism is the sacrament of regeneration and initiation."
    ref = "Baptism is the sacrament of regeneration into the Church."
    score = bleu_score(gen, ref)
    assert 0.0 < score < 1.0

    print("  ✓ bleu_score")


def test_rouge_l():
    from eval.metrics import rouge_l_score

    # Identical → perfect
    text = "The Trinity is one God in three persons."
    result = rouge_l_score(text, text)
    assert result["f1"] == 1.0

    # Completely different → 0
    result = rouge_l_score("hello", "goodbye forever")
    assert result["f1"] == 0.0

    # Empty input
    result = rouge_l_score("", "some text")
    assert result["f1"] == 0.0

    print("  ✓ rouge_l_score")


def test_token_f1():
    from eval.metrics import token_f1

    # Identical
    text = "Baptism forgives sins and initiates into the Church."
    result = token_f1(text, text)
    assert result["f1"] == 1.0

    # Partial overlap
    gen = "Baptism is a sacrament of the Church."
    ref = "Baptism is the first sacrament of initiation."
    result = token_f1(gen, ref)
    assert 0.0 < result["f1"] < 1.0

    print("  ✓ token_f1")


def test_source_coverage():
    from eval.metrics import source_coverage

    answer = (
        "According to the Catechism (CCC §1213), Baptism is the sacrament "
        "of regeneration. As St. Paul writes in Romans 6:3-4, we are buried "
        "with Christ through Baptism."
    )
    expected = ["CCC §1213", "CCC §1265", "Romans 6:3-4"]

    result = source_coverage(answer, expected)
    assert result["coverage"] == 2 / 3  # CCC §1265 is missing
    assert "CCC §1213" in result["cited"]
    assert "Romans 6:3-4" in result["cited"]
    assert "CCC §1265" in result["missing"]

    print("  ✓ source_coverage")


def test_score_single():
    from eval.metrics import score_single

    result = score_single(
        question="What is Baptism?",
        expected_answer="Baptism is the sacrament of regeneration.",
        expected_sources=["CCC §1213"],
        generated_answer="According to CCC §1213, Baptism is a sacrament of regeneration and new life.",
        retrieved_chunks=[
            {"reference": "CCC §1213", "paragraph_number": 1213},
            {"reference": "CCC §42", "paragraph_number": 42},
        ],
        abstained=False,
    )

    assert result["precision_at_k"] == 0.5
    assert result["recall_at_k"] == 1.0
    assert result["mrr"] == 1.0
    assert result["bleu"] >= 0  # BLEU can be 0 for short, divergent texts
    assert result["rouge_l_f1"] > 0  # ROUGE-L is more forgiving
    assert result["source_coverage"] == 1.0
    assert result["abstained"] is False

    print("  ✓ score_single")


def test_aggregate_scores():
    from eval.metrics import aggregate_scores

    results = [
        {
            "question": "Q1", "topic": "sacraments", "abstained": False,
            "precision_at_k": 0.5, "recall_at_k": 1.0, "mrr": 1.0,
            "bleu": 0.3, "rouge_l_f1": 0.5, "token_f1": 0.6,
            "source_coverage": 1.0,
        },
        {
            "question": "Q2", "topic": "creed", "abstained": False,
            "precision_at_k": 0.25, "recall_at_k": 0.5, "mrr": 0.5,
            "bleu": 0.2, "rouge_l_f1": 0.4, "token_f1": 0.5,
            "source_coverage": 0.5,
        },
    ]

    agg = aggregate_scores(results)
    assert agg["mean_precision_at_k"] == 0.375
    assert agg["total_questions"] == 2
    assert agg["abstention_rate"] == 0.0
    assert "by_topic" in agg

    print("  ✓ aggregate_scores")


# ── Runner ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n[eval/metrics] Running unit tests...\n")
    tests = [
        test_normalize_reference,
        test_chunk_matches_source,
        test_precision_at_k,
        test_recall_at_k,
        test_mrr,
        test_bleu_score,
        test_rouge_l,
        test_token_f1,
        test_source_coverage,
        test_score_single,
        test_aggregate_scores,
    ]

    failed = 0
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"  ✗ {test.__name__}: {e}")
            failed += 1

    print()
    if failed:
        print(f"  {failed} test(s) FAILED")
        sys.exit(1)
    else:
        print(f"  All {len(tests)} tests passed ✓")
        sys.exit(0)
