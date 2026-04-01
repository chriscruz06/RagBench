"""
Evaluation runner — score the full pipeline against the test set.

Feeds each question from test_set.json through retrieve → generate,
then scores the results using eval/metrics.py. Outputs a timestamped
JSON report to eval/results/.

Usage:
    python -m eval.runner                  # run full eval
    python -m eval.runner --retrieval-only # skip generation (faster, no Ollama needed)
    python -m eval.runner --top-k 10       # override top_k for this run
    python -m eval.runner --dry-run        # score test set with dummy answers (tests the harness)

The report includes:
  - Per-question scores (retrieval + generation + citation)
  - Aggregate means + per-topic breakdown
  - Pipeline config snapshot (embedding model, chunk strategy, top_k, etc.)
  - Timing info
"""

import json
import time
import argparse
from datetime import datetime
from pathlib import Path

from config.settings import settings
from eval.metrics import score_single, aggregate_scores


RESULTS_DIR = Path("eval/results")


def load_test_set(path: Path = None) -> list[dict]:
    """Load the test set from JSON."""
    path = path or settings.eval_test_set_path
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_pipeline_config() -> dict:
    """Snapshot the current pipeline configuration for the report."""
    return {
        "embedding_model": settings.embedding_model,
        "chunk_strategy": settings.chunk_strategy,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "top_k": settings.top_k,
        "similarity_threshold": settings.similarity_threshold,
        "llm_provider": settings.llm_provider,
        "ollama_model": settings.ollama_model if settings.llm_provider == "ollama" else None,
        "openai_model": settings.openai_model if settings.llm_provider == "openai" else None,
        "temperature": settings.temperature,
        "max_response_tokens": settings.max_response_tokens,
    }


def run_retrieval_only(question: str, top_k: int = None) -> dict:
    """
    Run only the retrieval step (no LLM call).
    Useful for fast iteration on embedding/chunking changes.
    """
    from retrieval.search import retrieve

    chunks = retrieve(question, top_k=top_k)
    return {
        "retrieved_chunks": chunks,
        "answer": "(retrieval-only mode — no generation)",
        "chunks_used": len(chunks),
        "abstained": len(chunks) == 0,
        "sources": [c.metadata for c in chunks],
    }


def run_full_pipeline(question: str, top_k: int = None) -> dict:
    """Run the full retrieve → generate pipeline."""
    from pipeline import query
    return query(question, top_k=top_k)


def evaluate(
    retrieval_only: bool = False,
    top_k: int = None,
    dry_run: bool = False,
    test_set_path: Path = None,
    tag: str = None,
) -> dict:
    """
    Run the full evaluation.

    Args:
        retrieval_only: skip generation, only score retrieval
        top_k: override top_k setting for this run
        dry_run: don't call the pipeline, use dummy data (tests the harness)
        test_set_path: override path to test set JSON
        tag: optional label for this run (e.g., "bge-base-rerank")

    Returns:
        Full report dict (also saved to eval/results/)
    """
    test_set = load_test_set(test_set_path)
    effective_top_k = top_k or settings.top_k

    print(f"\n[eval] Starting evaluation run")
    print(f"  Test questions: {len(test_set)}")
    print(f"  Mode: {'retrieval-only' if retrieval_only else 'full pipeline'}")
    print(f"  top_k: {effective_top_k}")
    if tag:
        print(f"  Tag: {tag}")
    print()

    per_question_results = []
    total_start = time.time()

    for i, test_case in enumerate(test_set, 1):
        question = test_case["question"]
        expected_answer = test_case["expected_answer"]
        expected_sources = test_case.get("expected_sources", [])
        topic = test_case.get("topic", "unknown")

        print(f"  [{i}/{len(test_set)}] {question[:60]}...", end=" ", flush=True)
        q_start = time.time()

        if dry_run:
            # Dummy result for testing the harness
            pipeline_result = {
                "answer": f"Dry run answer for: {question}",
                "chunks_used": 0,
                "abstained": True,
                "sources": [],
                "retrieved_chunks": [],
            }
        elif retrieval_only:
            pipeline_result = run_retrieval_only(question, top_k=effective_top_k)
        else:
            pipeline_result = run_full_pipeline(question, top_k=effective_top_k)

        q_elapsed = time.time() - q_start

        # Extract chunk metadata for scoring
        retrieved_chunks_meta = []
        raw_chunks = pipeline_result.get("retrieved_chunks", [])
        for chunk in raw_chunks:
            if isinstance(chunk, dict):
                retrieved_chunks_meta.append(chunk)
            else:
                # RetrievedChunk dataclass
                retrieved_chunks_meta.append(chunk.metadata)

        # Score this question
        scores = score_single(
            question=question,
            expected_answer=expected_answer,
            expected_sources=expected_sources,
            generated_answer=pipeline_result.get("answer", ""),
            retrieved_chunks=retrieved_chunks_meta,
            abstained=pipeline_result.get("abstained", False),
        )

        # Add metadata to the score dict
        scores["topic"] = topic
        scores["expected_answer"] = expected_answer
        scores["expected_sources"] = expected_sources
        scores["generated_answer"] = pipeline_result.get("answer", "")
        scores["chunks_used"] = pipeline_result.get("chunks_used", 0)
        scores["elapsed_seconds"] = round(q_elapsed, 2)

        per_question_results.append(scores)

        # Print compact result
        status = "ABSTAIN" if scores["abstained"] else "OK"
        print(f"[{status}] P@K={scores['precision_at_k']:.2f} "
              f"R@K={scores['recall_at_k']:.2f} "
              f"MRR={scores['mrr']:.2f} "
              f"F1={scores['token_f1']:.2f} "
              f"({q_elapsed:.1f}s)")

    total_elapsed = time.time() - total_start

    # Aggregate
    aggregate = aggregate_scores(per_question_results)

    # Build report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "meta": {
            "timestamp": datetime.now().isoformat(),
            "tag": tag,
            "mode": "retrieval-only" if retrieval_only else "full",
            "dry_run": dry_run,
            "total_elapsed_seconds": round(total_elapsed, 2),
            "test_set_size": len(test_set),
        },
        "config": get_pipeline_config(),
        "aggregate": aggregate,
        "per_question": per_question_results,
    }

    # Save report
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag_suffix = f"_{tag}" if tag else ""
    mode_suffix = "_retrieval" if retrieval_only else ""
    filename = f"eval_{timestamp}{tag_suffix}{mode_suffix}.json"
    report_path = RESULTS_DIR / filename

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"  EVALUATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Questions:        {aggregate.get('total_questions', 0)}")
    print(f"  Abstention rate:  {aggregate.get('abstention_rate', 0):.1%}")
    print(f"  Total time:       {total_elapsed:.1f}s")
    print(f"\n  Retrieval:")
    print(f"    Mean Precision@K:  {aggregate.get('mean_precision_at_k', 0):.4f}")
    print(f"    Mean Recall@K:     {aggregate.get('mean_recall_at_k', 0):.4f}")
    print(f"    Mean MRR:          {aggregate.get('mean_mrr', 0):.4f}")
    if not retrieval_only:
        print(f"\n  Generation:")
        print(f"    Mean BLEU:         {aggregate.get('mean_bleu', 0):.4f}")
        print(f"    Mean ROUGE-L F1:   {aggregate.get('mean_rouge_l_f1', 0):.4f}")
        print(f"    Mean Token F1:     {aggregate.get('mean_token_f1', 0):.4f}")
        print(f"    Mean Src Coverage: {aggregate.get('mean_source_coverage', 0):.4f}")

    if "by_topic" in aggregate:
        print(f"\n  By Topic:")
        for topic, stats in aggregate["by_topic"].items():
            print(f"    {topic:15s}  P@K={stats['precision_at_k']:.3f}  "
                  f"R@K={stats['recall_at_k']:.3f}  "
                  f"F1={stats['token_f1']:.3f}  "
                  f"(n={stats['count']})")

    print(f"\n  Report saved: {report_path}")
    print(f"{'=' * 60}\n")

    return report


# ── CLI ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RagBench Evaluation Runner")
    parser.add_argument(
        "--retrieval-only", action="store_true",
        help="Only evaluate retrieval (skip LLM generation)",
    )
    parser.add_argument(
        "--top-k", type=int, default=None,
        help="Override top_k for this run",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Test the harness without calling the pipeline",
    )
    parser.add_argument(
        "--tag", type=str, default=None,
        help="Label for this run (e.g., 'baseline', 'bge-rerank')",
    )
    parser.add_argument(
        "--test-set", type=str, default=None,
        help="Path to test set JSON (default: eval/test_set.json)",
    )

    args = parser.parse_args()

    evaluate(
        retrieval_only=args.retrieval_only,
        top_k=args.top_k,
        dry_run=args.dry_run,
        tag=args.tag,
        test_set_path=Path(args.test_set) if args.test_set else None,
    )
