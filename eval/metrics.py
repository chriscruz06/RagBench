"""
Evaluation metrics for retrieval and generation quality.

Retrieval metrics measure whether the right chunks are retrieved:
  - Precision@K: fraction of retrieved chunks that are relevant
  - Recall@K: fraction of relevant chunks that are retrieved
  - MRR (Mean Reciprocal Rank): how high the first relevant chunk ranks

Generation metrics measure answer quality:
  - BLEU: n-gram overlap with expected answer (precision-oriented)
  - ROUGE-L: longest common subsequence overlap (recall-oriented)
  - Token F1: bag-of-words overlap between generated and expected answer

Source coverage measures citation completeness:
  - What fraction of expected sources appear in the generated answer text

Usage:
    from eval.metrics import (
        precision_at_k, recall_at_k, mean_reciprocal_rank,
        bleu_score, rouge_l_score, token_f1,
        source_coverage,
    )
"""

import re
import math
from collections import Counter


# ═══════════════════════════════════════════════════════════════════
#  RETRIEVAL METRICS
# ═══════════════════════════════════════════════════════════════════

def _normalize_reference(ref: str) -> str:
    """
    Normalize a source reference for matching.

    Handles variations like:
      "CCC §1213"  → "ccc 1213"
      "CCC  §1213" → "ccc 1213"
      "Romans 6:3-4" → "romans 6:3-4"
      "1 Cor 13:4"   → "1 cor 13:4"
    """
    ref = ref.lower().strip()
    # Remove § and ¶ symbols
    ref = ref.replace("§", "").replace("¶", "")
    # Collapse whitespace
    ref = re.sub(r"\s+", " ", ref)
    return ref


def _chunk_matches_source(chunk_metadata: dict, expected_source: str) -> bool:
    """
    Check if a retrieved chunk matches an expected source reference.

    Matching strategy:
    1. Direct match on the chunk's 'reference' field (e.g., "CCC §1213")
    2. Paragraph number match for CCC (e.g., expected "CCC §1213" matches
       a chunk with paragraph_number=1213)
    3. Substring match on source filename for Bible references
       (e.g., "Romans 6:3-4" matches source="romans.txt" or text containing
       "Romans 6")
    """
    norm_expected = _normalize_reference(expected_source)

    # Strategy 1: Direct reference match
    chunk_ref = chunk_metadata.get("reference", "")
    if chunk_ref and _normalize_reference(chunk_ref) == norm_expected:
        return True

    # Strategy 2: CCC paragraph number match
    ccc_match = re.search(r"ccc\s*(\d+)", norm_expected)
    if ccc_match:
        expected_para = int(ccc_match.group(1))
        chunk_para = chunk_metadata.get("paragraph_number")
        if chunk_para is not None and int(chunk_para) == expected_para:
            return True

    # Strategy 3: For scripture refs, check if the chunk's source file
    # or text relates to the expected book
    # Extract book name from expected source (e.g., "Romans" from "Romans 6:3-4")
    scripture_match = re.match(
        r"(\d?\s*[a-z]+)",
        norm_expected,
    )
    if scripture_match and "ccc" not in norm_expected:
        book_name = scripture_match.group(1).strip()
        chunk_source = chunk_metadata.get("source", "").lower()
        # Check if the book name appears in the chunk's source filename
        # e.g., "romans.txt" contains "romans"
        if book_name and book_name in chunk_source:
            return True

    return False


def precision_at_k(
    retrieved_chunks: list[dict],
    expected_sources: list[str],
) -> float:
    """
    Precision@K: Of the K chunks retrieved, how many are relevant?

    Args:
        retrieved_chunks: list of chunk metadata dicts from retrieval
        expected_sources: list of expected source references

    Returns:
        float in [0, 1]
    """
    if not retrieved_chunks:
        return 0.0

    relevant_count = sum(
        1 for chunk in retrieved_chunks
        if any(_chunk_matches_source(chunk, src) for src in expected_sources)
    )

    return relevant_count / len(retrieved_chunks)


def recall_at_k(
    retrieved_chunks: list[dict],
    expected_sources: list[str],
) -> float:
    """
    Recall@K: Of the expected sources, how many were retrieved?

    Args:
        retrieved_chunks: list of chunk metadata dicts from retrieval
        expected_sources: list of expected source references

    Returns:
        float in [0, 1]
    """
    if not expected_sources:
        return 1.0  # nothing to recall

    found = set()
    for src in expected_sources:
        for chunk in retrieved_chunks:
            if _chunk_matches_source(chunk, src):
                found.add(src)
                break

    return len(found) / len(expected_sources)


def mean_reciprocal_rank(
    retrieved_chunks: list[dict],
    expected_sources: list[str],
) -> float:
    """
    MRR: Reciprocal of the rank of the first relevant chunk.

    If the first relevant chunk is rank 1, MRR = 1.0.
    If rank 3, MRR = 0.333. If no relevant chunk found, MRR = 0.0.

    Args:
        retrieved_chunks: list of chunk metadata dicts, in retrieval order
        expected_sources: list of expected source references

    Returns:
        float in [0, 1]
    """
    for rank, chunk in enumerate(retrieved_chunks, 1):
        if any(_chunk_matches_source(chunk, src) for src in expected_sources):
            return 1.0 / rank

    return 0.0


# ═══════════════════════════════════════════════════════════════════
#  GENERATION METRICS
# ═══════════════════════════════════════════════════════════════════

def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer for metric computation."""
    text = text.lower()
    # Split on non-alphanumeric characters, keep tokens
    tokens = re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text)
    return tokens


def _get_ngrams(tokens: list[str], n: int) -> Counter:
    """Extract n-grams from a token list."""
    return Counter(
        tuple(tokens[i : i + n])
        for i in range(len(tokens) - n + 1)
    )


def bleu_score(
    generated: str,
    reference: str,
    max_n: int = 4,
) -> float:
    """
    Corpus-level BLEU score (single reference).

    Uses modified precision with brevity penalty, up to max_n-grams.
    This is a simplified implementation suitable for single-reference eval.

    Args:
        generated: the model's generated answer
        reference: the expected/gold answer
        max_n: maximum n-gram order (default 4)

    Returns:
        float in [0, 1]
    """
    gen_tokens = _tokenize(generated)
    ref_tokens = _tokenize(reference)

    if not gen_tokens or not ref_tokens:
        return 0.0

    # Clipped n-gram precisions
    precisions = []
    for n in range(1, max_n + 1):
        gen_ngrams = _get_ngrams(gen_tokens, n)
        ref_ngrams = _get_ngrams(ref_tokens, n)

        if not gen_ngrams:
            precisions.append(0.0)
            continue

        clipped_count = 0
        for ngram, count in gen_ngrams.items():
            clipped_count += min(count, ref_ngrams.get(ngram, 0))

        precision = clipped_count / sum(gen_ngrams.values())
        precisions.append(precision)

    # If any precision is 0, BLEU is 0 (log would be -inf)
    if any(p == 0 for p in precisions):
        return 0.0

    # Geometric mean of precisions
    log_avg = sum(math.log(p) for p in precisions) / len(precisions)

    # Brevity penalty
    bp = 1.0
    if len(gen_tokens) < len(ref_tokens):
        bp = math.exp(1 - len(ref_tokens) / len(gen_tokens))

    return bp * math.exp(log_avg)


def rouge_l_score(generated: str, reference: str) -> dict:
    """
    ROUGE-L: Longest Common Subsequence based metric.

    Returns precision, recall, and F1 based on LCS length.

    Args:
        generated: the model's generated answer
        reference: the expected/gold answer

    Returns:
        dict with keys "precision", "recall", "f1"
    """
    gen_tokens = _tokenize(generated)
    ref_tokens = _tokenize(reference)

    if not gen_tokens or not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # Compute LCS length using DP
    m, n = len(gen_tokens), len(ref_tokens)
    # Use 1D DP to save memory
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if gen_tokens[i - 1] == ref_tokens[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev = curr

    lcs_len = prev[n]

    precision = lcs_len / m if m > 0 else 0.0
    recall = lcs_len / n if n > 0 else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {"precision": precision, "recall": recall, "f1": f1}


def token_f1(generated: str, reference: str) -> dict:
    """
    Token-level F1: bag-of-words overlap between generated and reference.

    This is a simpler, more forgiving metric than BLEU — it doesn't care
    about word order, just whether the right words appear.

    Args:
        generated: the model's generated answer
        reference: the expected/gold answer

    Returns:
        dict with keys "precision", "recall", "f1"
    """
    gen_tokens = set(_tokenize(generated))
    ref_tokens = set(_tokenize(reference))

    if not gen_tokens or not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    overlap = gen_tokens & ref_tokens

    precision = len(overlap) / len(gen_tokens) if gen_tokens else 0.0
    recall = len(overlap) / len(ref_tokens) if ref_tokens else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {"precision": precision, "recall": recall, "f1": f1}


# ═══════════════════════════════════════════════════════════════════
#  SOURCE COVERAGE (citation quality)
# ═══════════════════════════════════════════════════════════════════

def source_coverage(
    answer_text: str,
    expected_sources: list[str],
) -> dict:
    """
    Check which expected sources are explicitly cited in the generated answer.

    This checks the answer text for CCC paragraph references and scripture
    citations, measuring whether the model actually cited the right sources.

    Args:
        answer_text: the generated answer
        expected_sources: list of expected source references

    Returns:
        dict with "coverage" (float), "cited" (list), "missing" (list)
    """
    if not expected_sources:
        return {"coverage": 1.0, "cited": [], "missing": []}

    answer_lower = answer_text.lower()

    cited = []
    missing = []

    for src in expected_sources:
        # Build search patterns for this source
        found = False
        norm = _normalize_reference(src)

        # Check for CCC references: "CCC §1213", "§1213", "paragraph 1213"
        ccc_match = re.search(r"ccc\s*(\d+)", norm)
        if ccc_match:
            para_num = ccc_match.group(1)
            patterns = [
                f"§{para_num}",
                f"§ {para_num}",
                f"ccc {para_num}",
                f"ccc§{para_num}",
                f"paragraph {para_num}",
            ]
            for pattern in patterns:
                if pattern in answer_lower:
                    found = True
                    break

        # Check for scripture references: "Romans 6:3", "Rom 6:3"
        if not found and "ccc" not in norm:
            # Try the reference as-is (case-insensitive)
            if norm in answer_lower:
                found = True
            else:
                # Try just the book name + chapter
                chapter_match = re.match(r"(.+?)\s+(\d+)", norm)
                if chapter_match:
                    book_chapter = chapter_match.group(0)
                    if book_chapter in answer_lower:
                        found = True

        if found:
            cited.append(src)
        else:
            missing.append(src)

    coverage = len(cited) / len(expected_sources)
    return {"coverage": coverage, "cited": cited, "missing": missing}


# ═══════════════════════════════════════════════════════════════════
#  AGGREGATE SCORING
# ═══════════════════════════════════════════════════════════════════

def score_single(
    question: str,
    expected_answer: str,
    expected_sources: list[str],
    generated_answer: str,
    retrieved_chunks: list[dict],
    abstained: bool = False,
) -> dict:
    """
    Compute all metrics for a single question-answer pair.

    This is the main entry point for scoring — it runs every metric
    and returns a flat dict of results.

    Args:
        question: the input question
        expected_answer: gold-standard answer
        expected_sources: list of expected source references
        generated_answer: the model's generated answer
        retrieved_chunks: list of chunk metadata dicts from retrieval
        abstained: whether the pipeline abstained from answering

    Returns:
        dict with all metric values
    """
    # Retrieval metrics
    p_at_k = precision_at_k(retrieved_chunks, expected_sources)
    r_at_k = recall_at_k(retrieved_chunks, expected_sources)
    mrr = mean_reciprocal_rank(retrieved_chunks, expected_sources)

    # Generation metrics (skip if abstained)
    if abstained:
        bleu = 0.0
        rouge = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        f1 = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        coverage = {"coverage": 0.0, "cited": [], "missing": expected_sources}
    else:
        bleu = bleu_score(generated_answer, expected_answer)
        rouge = rouge_l_score(generated_answer, expected_answer)
        f1 = token_f1(generated_answer, expected_answer)
        coverage = source_coverage(generated_answer, expected_sources)

    return {
        "question": question,
        "abstained": abstained,
        # Retrieval
        "precision_at_k": round(p_at_k, 4),
        "recall_at_k": round(r_at_k, 4),
        "mrr": round(mrr, 4),
        # Generation
        "bleu": round(bleu, 4),
        "rouge_l_f1": round(rouge["f1"], 4),
        "rouge_l_precision": round(rouge["precision"], 4),
        "rouge_l_recall": round(rouge["recall"], 4),
        "token_f1": round(f1["f1"], 4),
        "token_precision": round(f1["precision"], 4),
        "token_recall": round(f1["recall"], 4),
        # Citation quality
        "source_coverage": round(coverage["coverage"], 4),
        "sources_cited": coverage["cited"],
        "sources_missing": coverage["missing"],
    }


def aggregate_scores(results: list[dict]) -> dict:
    """
    Compute aggregate statistics across all test questions.

    Args:
        results: list of score dicts from score_single()

    Returns:
        dict with mean values for each metric + per-topic breakdown
    """
    if not results:
        return {}

    # Numeric metric keys to aggregate
    metric_keys = [
        "precision_at_k", "recall_at_k", "mrr",
        "bleu", "rouge_l_f1", "token_f1", "source_coverage",
    ]

    # Overall means
    aggregate = {}
    for key in metric_keys:
        values = [r[key] for r in results if key in r]
        aggregate[f"mean_{key}"] = round(sum(values) / len(values), 4) if values else 0.0

    # Abstention rate
    abstained_count = sum(1 for r in results if r.get("abstained", False))
    aggregate["abstention_rate"] = round(abstained_count / len(results), 4)
    aggregate["total_questions"] = len(results)

    # Per-topic breakdown (if topic info is available)
    topics = {}
    for r in results:
        topic = r.get("topic", "unknown")
        if topic not in topics:
            topics[topic] = []
        topics[topic].append(r)

    if len(topics) > 1:
        topic_breakdown = {}
        for topic, topic_results in topics.items():
            topic_agg = {}
            for key in metric_keys:
                values = [r[key] for r in topic_results if key in r]
                topic_agg[key] = round(sum(values) / len(values), 4) if values else 0.0
            topic_agg["count"] = len(topic_results)
            topic_breakdown[topic] = topic_agg
        aggregate["by_topic"] = topic_breakdown

    return aggregate
