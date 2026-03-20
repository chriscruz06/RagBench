"""
RagBench configuration — all knobs in one place.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path


class Settings(BaseSettings):
    # ── Paths ──────────────────────────────────────────────
    project_root: Path = Path(__file__).resolve().parent.parent
    raw_data_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "raw")
    processed_data_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "processed")
    chroma_persist_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "chroma_db")

    # ── Embedding ──────────────────────────────────────────
    embedding_model: str = "all-MiniLM-L6-v2"  # fast, good baseline
    # upgrade path: "BAAI/bge-base-en-v1.5" for better quality

    # ── Chunking ──────────────────────────────────────────
    chunk_strategy: str = "fixed"  # "fixed" | "sentence" | "semantic"
    chunk_size: int = 512  # tokens (for fixed strategy)
    chunk_overlap: int = 64

    # ── Retrieval ─────────────────────────────────────────
    top_k: int = 5
    similarity_threshold: float = 0.3  # below this → abstain

    # ── LLM ───────────────────────────────────────────────
    llm_provider: str = "ollama"  # "ollama" | "openai"
    ollama_model: str = "mistral"  # good balance of speed and quality
    ollama_base_url: str = "http://localhost:11434"
    openai_model: str = "gpt-4o-mini"
    openai_api_key: str = ""

    # ── Generation ────────────────────────────────────────
    max_response_tokens: int = 1024
    temperature: float = 0.1  # low temp = less hallucination

    # ── Eval ──────────────────────────────────────────────
    eval_test_set_path: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "eval" / "test_set.json")

    model_config = {"env_prefix": "RAGBENCH_", "env_file": ".env"}


settings = Settings()
