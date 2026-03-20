"""
Document loading — read raw files into a unified Document format.

Supports: .pdf, .txt, .md
Each document gets tagged with metadata (source, doc_type) so we can
trace retrieval results back to "Bible" vs "Catechism" in the UI.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Document:
    """Single unit of ingested text with provenance metadata."""
    text: str
    metadata: dict = field(default_factory=dict)
    # metadata keys we care about:
    #   source: str        — filename or URL
    #   doc_type: str      — "bible" | "catechism" | "commentary"
    #   reference: str     — e.g. "Genesis 1:1" or "CCC §1234"


def load_text_file(path: Path, doc_type: str = "unknown") -> list[Document]:
    """Load a plain text or markdown file into Documents (one per file)."""
    text = path.read_text(encoding="utf-8")
    return [Document(
        text=text,
        metadata={"source": path.name, "doc_type": doc_type}
    )]


def load_pdf_file(path: Path, doc_type: str = "unknown") -> list[Document]:
    """Load a PDF file — one Document per page."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    docs = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            docs.append(Document(
                text=text,
                metadata={
                    "source": path.name,
                    "doc_type": doc_type,
                    "page": i + 1,
                }
            ))
    return docs


def load_json_file(path: Path, doc_type: str = "unknown") -> list[Document]:
    """
    Load a JSON file of pre-parsed paragraphs (e.g., CCC output).
    """
    import json

    with open(path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    docs = []
    for entry in entries:
        if isinstance(entry, dict) and entry.get("text"):
            metadata = {
                "source": path.name,
                "doc_type": doc_type,
            }
            for key in ("reference", "paragraph_number"):
                if key in entry:
                    metadata[key] = entry[key]
            # ChromaDB can't store empty lists, convert to comma-separated string
            if entry.get("scripture_refs"):
                metadata["scripture_refs"] = ", ".join(entry["scripture_refs"])

            docs.append(Document(text=entry["text"], metadata=metadata))

    return docs

LOADERS = {
    ".txt": load_text_file,
    ".md": load_text_file,
    ".pdf": load_pdf_file,
    ".json": load_json_file,
}


def load_directory(dir_path: Path, doc_type: str = "unknown") -> list[Document]:
    """
    Load all supported files from a directory.

    Usage:
        docs = load_directory(Path("data/raw/bible"), doc_type="bible")
    """
    dir_path = Path(dir_path)
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    documents = []
    for file_path in sorted(dir_path.iterdir()):
        loader = LOADERS.get(file_path.suffix.lower())
        if loader:
            documents.extend(loader(file_path, doc_type=doc_type))

    print(f"[ingestion] Loaded {len(documents)} documents from {dir_path}")
    return documents
