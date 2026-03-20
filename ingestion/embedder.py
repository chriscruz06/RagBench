"""
Embedding and vector store — embed chunks and persist to ChromaDB.
"""

from pathlib import Path
from ingestion.loader import Document
from config.settings import settings


def get_embedding_function():
    """Return the embedding function for ChromaDB."""
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)


def store_chunks(chunks: list[Document], collection_name: str = "ragbench") -> None:
    """
    Embed chunks and store them in ChromaDB.

    Each chunk gets a deterministic ID based on source + chunk index
    so re-ingestion is idempotent (won't create duplicates).
    """
    import chromadb

    persist_dir = str(settings.chroma_persist_dir)
    client = chromadb.PersistentClient(path=persist_dir)

    embedding_fn = get_embedding_function()

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    # Prepare batch
    ids = []
    texts = []
    metadatas = []

    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "unknown")
        para_num = chunk.metadata.get("paragraph_number", "")
        chunk_idx = chunk.metadata.get("chunk_index", i)
        chunk_id = f"{source}::p{para_num}::chunk_{chunk_idx}_{i}"

        ids.append(chunk_id)
        texts.append(chunk.text)
        metadatas.append(chunk.metadata)

    # Embed and upsert
    embeddings = embedding_fn.embed_documents(texts)

    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print(f"[embedder] Stored {len(chunks)} chunks in collection '{collection_name}'")


def get_collection(collection_name: str = "ragbench"):
    """Get a handle to the ChromaDB collection for querying."""
    import chromadb

    persist_dir = str(settings.chroma_persist_dir)
    client = chromadb.PersistentClient(path=persist_dir)
    return client.get_collection(name=collection_name)
