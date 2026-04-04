"""
Chunking ablation experiment - Phase 3.

Re-ingests the full corpus under each chunking strategy (fixed, sentence,
semantic), runs the eval suite for each, and saves tagged reports for
comparison with eval/report.py.

This is the centerpiece of Phase 3: controlled experiments that isolate
the effect of chunking strategy on retrieval and generation quality.

Usage:
    python -m experiments.ablation                          # full ablation (all 3 strategies)
    python -m experiments.ablation --strategies fixed sentence  # subset
    python -m experiments.ablation --retrieval-only         # skip LLM (faster)
    python -m experiments.ablation --dry-run                # test harness only

After running:
    python -m eval.report --all                             # compare all runs
    python -m eval.report --diff                            # side-by-side deltas
    python -m eval.report --baseline chunk_fixed --latest 3 # vs. fixed baseline

Interview talking point: "I ran controlled ablation experiments across three
chunking strategies, holding embedding model, retrieval top-K, and reranker
constant. Each run re-ingested ~7,500 vectors from scratch and evaluated
against 10 hand-crafted theological QA triples."
"""

import time
import argparse
import chromadb
from pathlib import Path

from config.settings import settings
from eval.runner import evaluate


# ── Strategies to ablate ──

ALL_STRATEGIES = ["fixed", "sentence", "semantic"]

# Corpus directories (processed, ready for ingestion)
CORPUS_DIRS = [
    ("data/processed/bible", "bible"),
    ("data/processed/catechism", "catechism"),
]


def reset_collection(collection_name: str = "ragbench") -> None:
    """
    Delete and recreate the ChromaDB collection.

    This ensures each ablation run starts from a clean slate,
    no leftover vectors from a previous chunking strategy.
    """
    persist_dir = str(settings.chroma_persist_dir)
    client = chromadb.PersistentClient(path=persist_dir)

    try:
        client.delete_collection(collection_name)
        print(f"  [reset] Deleted collection '{collection_name}'")
    except ValueError:
        print(f"  [reset] Collection '{collection_name}' not found (clean start)")

    # Recreate empty collection with cosine similarity
    client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    print(f"  [reset] Created fresh collection '{collection_name}'")


def ingest_with_strategy(strategy: str) -> dict:
    """
    Re-ingest the full corpus using a specific chunking strategy.

    Returns summary stats: {documents_loaded, chunks_stored, elapsed}.
    """
    from ingestion.loader import load_directory
    from ingestion.chunker import chunk_documents
    from ingestion.embedder import store_chunks

    total_docs = 0
    total_chunks = 0
    start = time.time()

    for data_dir, doc_type in CORPUS_DIRS:
        path = Path(data_dir)
        if not path.exists():
            print(f"  [WARN] Corpus not found: {path} - skipping")
            continue

        docs = load_directory(path, doc_type=doc_type)
        chunks = chunk_documents(docs, strategy=strategy)
        store_chunks(chunks)

        total_docs += len(docs)
        total_chunks += len(chunks)

    elapsed = time.time() - start

    return {
        "documents_loaded": total_docs,
        "chunks_stored": total_chunks,
        "ingestion_seconds": round(elapsed, 2),
    }


def run_ablation(
    strategies: list[str] = None,
    retrieval_only: bool = False,
    dry_run: bool = False,
    top_k: int = None,
) -> list[dict]:
    """
    Run the full ablation experiment.

    For each strategy:
    1. Reset the vector store
    2. Re-ingest corpus with that strategy
    3. Run eval and save a tagged report

    Args:
        strategies: list of strategies to test (default: all three)
        retrieval_only: skip LLM generation (faster iteration)
        dry_run: skip pipeline calls (test the harness)
        top_k: override retrieval top_k

    Returns:
        List of eval reports (one per strategy).
    """
    strategies = strategies or ALL_STRATEGIES
    reports = []

    print("=" * 60)
    print("  CHUNKING ABLATION EXPERIMENT")
    print("=" * 60)
    print(f"  Strategies:  {', '.join(strategies)}")
    print(f"  Mode:        {'retrieval-only' if retrieval_only else 'full pipeline'}")
    print(f"  Embedding:   {settings.embedding_model}")
    print(f"  Chunk size:  {settings.chunk_size} (fixed/sentence)")
    if not dry_run:
        print(f"  Corpus dirs: {[d for d, _ in CORPUS_DIRS]}")
    print(f"{'=' * 60}\n")

    experiment_start = time.time()

    for i, strategy in enumerate(strategies, 1):
        tag = f"chunk_{strategy}"

        print(f"\n{'─' * 60}")
        print(f"  [{i}/{len(strategies)}] Strategy: {strategy} (tag: {tag})")
        print(f"{'─' * 60}")

        # ── Step 1: Reset vector store ────────────────────
        if not dry_run:
            reset_collection()
        else:
            print("  [dry-run] Skipping collection reset")

        # ── Step 2: Re-ingest with this strategy ──────────
        if not dry_run:
            # Temporarily override the chunk_strategy setting so the
            # eval report's config snapshot reflects what we actually ran
            original_strategy = settings.chunk_strategy
            settings.chunk_strategy = strategy

            print(f"\n  Ingesting with strategy='{strategy}'...")
            ingest_stats = ingest_with_strategy(strategy)
            print(f"  Ingestion complete: {ingest_stats['chunks_stored']} chunks "
                  f"in {ingest_stats['ingestion_seconds']:.1f}s")
        else:
            print("  [dry-run] Skipping ingestion")
            original_strategy = settings.chunk_strategy
            settings.chunk_strategy = strategy

        # ── Step 3: Run eval ──
        print(f"\n  Running evaluation (tag={tag})...")
        report = evaluate(
            retrieval_only=retrieval_only,
            top_k=top_k,
            dry_run=dry_run,
            tag=tag,
        )

        # Attach ingestion stats to the report
        if not dry_run:
            report["ingestion"] = ingest_stats

        reports.append(report)

        # Restore original setting
        settings.chunk_strategy = original_strategy

    total_elapsed = time.time() - experiment_start

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print(f"  ABLATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Total time: {total_elapsed:.1f}s")
    print(f"  Runs: {len(reports)}")
    print()

    # Compact comparison table
    header = f"  {'Strategy':<12} {'P@K':>6} {'R@K':>6} {'MRR':>6}"
    if not retrieval_only:
        header += f" {'F1':>6} {'ROUGE':>6} {'BLEU':>6} {'SrcCov':>6}"
    header += f" {'Chunks':>7}"
    print(header)
    print(f"  {'─' * (len(header) - 2)}")

    for report in reports:
        tag = report["meta"]["tag"]
        agg = report["aggregate"]
        strategy_name = tag.replace("chunk_", "")
        ingestion = report.get("ingestion", {})
        chunks = ingestion.get("chunks_stored", "?")

        row = (f"  {strategy_name:<12} "
               f"{agg.get('mean_precision_at_k', 0):>6.3f} "
               f"{agg.get('mean_recall_at_k', 0):>6.3f} "
               f"{agg.get('mean_mrr', 0):>6.3f}")
        if not retrieval_only:
            row += (f" {agg.get('mean_token_f1', 0):>6.3f}"
                    f" {agg.get('mean_rouge_l_f1', 0):>6.3f}"
                    f" {agg.get('mean_bleu', 0):>6.3f}"
                    f" {agg.get('mean_source_coverage', 0):>6.3f}")
        row += f" {str(chunks):>7}"
        print(row)

    print(f"\n  Reports saved to eval/results/")
    print(f"  Compare with: python -m eval.report --all")
    print(f"{'=' * 60}\n")

    return reports


# ── CLI ──

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RagBench Chunking Ablation Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python -m experiments.ablation                            # full ablation
  python -m experiments.ablation --strategies fixed sentence  # skip semantic
  python -m experiments.ablation --retrieval-only           # fast mode
  python -m experiments.ablation --dry-run                  # test harness

After running:
  python -m eval.report --all                               # compare results
  python -m eval.report --baseline chunk_fixed --latest 3   # vs. baseline
        """,
    )
    parser.add_argument(
        "--strategies", nargs="+", default=None,
        choices=ALL_STRATEGIES,
        help="Which strategies to test (default: all three)",
    )
    parser.add_argument(
        "--retrieval-only", action="store_true",
        help="Skip LLM generation, only evaluate retrieval metrics",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Test the harness without calling the pipeline",
    )
    parser.add_argument(
        "--top-k", type=int, default=None,
        help="Override retrieval top_k for all runs",
    )

    args = parser.parse_args()

    run_ablation(
        strategies=args.strategies,
        retrieval_only=args.retrieval_only,
        dry_run=args.dry_run,
        top_k=args.top_k,
    )
