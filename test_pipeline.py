"""
Verify each pipeline component works independently.

Usage:
    python test_pipeline.py

This is NOT the eval framework (that's Phase 2). This just checks that
the pipe works: documents load, chunks embed, retrieval returns
results, and generation produces an answer thats at least semi coherent
"""

import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

console = Console()


def test_loader():
    """Test that processed files can be loaded."""
    from ingestion.loader import load_directory

    bible_dir = Path("data/processed/bible")
    catechism_dir = Path("data/processed/catechism")

    errors = []
    if not bible_dir.exists():
        errors.append(f"Bible directory not found: {bible_dir}")
    else:
        docs = load_directory(bible_dir, doc_type="bible")
        if len(docs) == 0:
            errors.append("Bible loaded 0 documents")
        else:
            console.print(f"  ✓ Bible: {len(docs)} documents loaded")

    if not catechism_dir.exists():
        errors.append(f"Catechism directory not found: {catechism_dir}")
    else:
        docs = load_directory(catechism_dir, doc_type="catechism")
        if len(docs) == 0:
            errors.append("Catechism loaded 0 documents")
        else:
            console.print(f"  ✓ Catechism: {len(docs)} documents loaded")

    return errors


def test_chunker():
    """Test that chunking strategies work."""
    from ingestion.loader import Document
    from ingestion.chunker import chunk_documents

    sample = Document(
        text="God is love. " * 100,  # ~1300 chars
        metadata={"source": "test", "doc_type": "test"}
    )

    errors = []
    for strategy in ["fixed", "sentence"]:
        chunks = chunk_documents([sample], strategy=strategy)
        if len(chunks) == 0:
            errors.append(f"Strategy '{strategy}' produced 0 chunks")
        else:
            console.print(f"  ✓ {strategy} chunking: {len(chunks)} chunks")

    return errors


def test_retrieval():
    """Test that ChromaDB collection exists and retrieval works."""
    from retrieval.search import retrieve

    errors = []
    try:
        chunks = retrieve("What is baptism?", top_k=3)
        console.print(f"   Retrieval: {len(chunks)} chunks returned")
        for i, chunk in enumerate(chunks, 1):
            source = chunk.metadata.get("doc_type", "?")
            console.print(f"    [{i}] score={chunk.score:.3f} type={source} "
                          f"({chunk.text[:60]}...)")
    except Exception as e:
        errors.append(f"Retrieval failed: {e}")

    return errors


def test_generation():
    """Test that LLM generation works (requires Ollama running)."""
    from retrieval.search import retrieve
    from generation.generate import generate

    errors = []
    try:
        chunks = retrieve("What is the Trinity?", top_k=3)
        if not chunks:
            errors.append("No chunks retrieved, can't test generation")
            return errors

        result = generate("What is the Trinity?", chunks)

        if result["abstained"]:
            errors.append("Generation abstained unexpectedly")
        elif len(result["answer"]) < 50:
            errors.append(f"Answer suspiciously short: {result['answer']}")
        else:
            console.print(f"  ✓ Generation: {len(result['answer'])} chars, "
                          f"{result['chunks_used']} chunks used")
            console.print(f"    Preview: {result['answer'][:120]}...")

    except Exception as e:
        errors.append(f"Generation failed: {e}")

    return errors


if __name__ == "__main__":
    console.print(Panel("[bold]RagBench Smoke Test[/bold]", style="blue"))

    all_errors = []

    tests = [
        ("Document Loading", test_loader),
        ("Chunking", test_chunker),
        ("Retrieval", test_retrieval),
        ("Generation (requires Ollama)", test_generation),
    ]

    for name, test_fn in tests:
        console.print(f"\n[bold]{name}[/bold]")
        try:
            errors = test_fn()
            all_errors.extend(errors)
            for err in errors:
                console.print(f"  ✗ {err}", style="red")
        except Exception as e:
            console.print(f"  ✗ Unexpected error: {e}", style="red")
            all_errors.append(str(e))

    # Summary
    console.print()
    if all_errors:
        console.print(Panel(
            f"[red]{len(all_errors)} issue(s) found[/red]",
            title="Result",
        ))
        sys.exit(1)
    else:
        console.print(Panel(
            "[green]All checks passed ✓[/green]",
            title="Result",
        ))
        sys.exit(0)
