"""
RagBench pipeline — the end-to-end orchestrator.

Usage:
    from pipeline import ingest, query

    # Ingest documents
    ingest("data/raw/bible", doc_type="bible")
    ingest("data/raw/catechism", doc_type="catechism")

    # Query
    result = query("What does the Church teach about baptism?")
    print(result["answer"])
"""

from pathlib import Path

from ingestion.loader import load_directory
from ingestion.chunker import chunk_documents
from ingestion.embedder import store_chunks
from retrieval.search import retrieve
from generation.generate import generate


def ingest(data_dir: str, doc_type: str = "unknown", chunk_strategy: str = None):
    """
    Full ingestion pipeline: load → chunk → embed → store.
    """
    # 1. Load raw documents
    docs = load_directory(Path(data_dir), doc_type=doc_type)

    # 2. Chunk them
    chunks = chunk_documents(docs, strategy=chunk_strategy)

    # 3. Embed and store in ChromaDB
    store_chunks(chunks)

    return {"documents_loaded": len(docs), "chunks_stored": len(chunks)}


def query(question: str, top_k: int = None) -> dict:
    """
    Full query pipeline: retrieve → generate.

    Returns:
        {
            "question": str,
            "answer": str,
            "chunks_used": int,
            "abstained": bool,
            "sources": list[dict],
            "retrieved_chunks": list[RetrievedChunk],
        }
    """
    # 1. Retrieve relevant chunks
    chunks = retrieve(question, top_k=top_k)

    # 2. Generate grounded answer
    result = generate(question, chunks)

    return {
        "question": question,
        **result,
        "retrieved_chunks": chunks,
    }


if __name__ == "__main__":
    import sys
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown

    console = Console()

    if len(sys.argv) > 1 and sys.argv[1] == "ingest":
        # python pipeline.py ingest
        console.print("[bold green]Ingesting documents...[/bold green]")
        # Ingest each corpus layer
        for sub in ["bible", "catechism"]:
            data_path = Path("data/raw") / sub
            if data_path.exists():
                result = ingest(str(data_path), doc_type=sub)
                console.print(f"  {sub}: {result}")
            else:
                console.print(f"  [yellow]Skipping {sub} — {data_path} not found[/yellow]")
    else:
        # Interactive query mode
        console.print(Panel("RagBench — Theological Study Assistant", style="bold blue"))
        console.print("Ask a question (or 'quit' to exit):\n")

        while True:
            question = input("❯ ").strip()
            if question.lower() in ("quit", "exit", "q"):
                break
            if not question:
                continue

            result = query(question)
            console.print(Panel(
                Markdown(result["answer"]),
                title="Answer",
                subtitle=f"Sources: {result['chunks_used']} chunks | Abstained: {result['abstained']}",
            ))
