"""
Vector store management (Chroma).

We keep one persistent Chroma collection per embedding model so the embedding
comparison can be run without re-indexing or models clobbering each other.
"""

from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document

import config
from src.embeddings import get_embeddings


def _persist_dir(embedding_name: str) -> str:
    """Each embedding model gets its own on-disk directory."""
    return str(Path(config.CHROMA_DIR) / embedding_name)


def _collection_name(embedding_name: str) -> str:
    return f"papers_{embedding_name}"


def build_vectorstore(
    chunks: List[Document], embedding_name: str = config.DEFAULT_EMBEDDING
) -> Chroma:
    """(Re)build and persist a Chroma collection for the given embedding model."""
    embeddings = get_embeddings(embedding_name)
    store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=_persist_dir(embedding_name),
        collection_name=_collection_name(embedding_name),
    )
    return store


def add_to_vectorstore(
    chunks: List[Document], embedding_name: str = config.DEFAULT_EMBEDDING
) -> Chroma:
    """Append chunks to an existing collection (used for live document uploads)."""
    store = load_vectorstore(embedding_name)
    if chunks:
        store.add_documents(chunks)
    return store


def has_vectorstore(embedding_name: str) -> bool:
    return Path(_persist_dir(embedding_name)).exists()


def collection_stats(embedding_name: str) -> dict:
    """Read vector count + dimension straight from the persisted Chroma
    collection (no embedding model load) — concrete proof of what's indexed."""
    import chromadb

    try:
        client = chromadb.PersistentClient(path=_persist_dir(embedding_name))
        col = client.get_collection(_collection_name(embedding_name))
        count = col.count()
        sample = col.get(limit=1, include=["embeddings"])
        embs = sample.get("embeddings")  # may be a numpy array — avoid bool coercion
        dim = len(embs[0]) if embs is not None and len(embs) > 0 else None
        return {"collection": _collection_name(embedding_name), "vectors": count, "dim": dim}
    except Exception:
        return {"collection": _collection_name(embedding_name), "vectors": None, "dim": None}


def load_vectorstore(embedding_name: str = config.DEFAULT_EMBEDDING) -> Chroma:
    """Load an already-built Chroma collection."""
    persist_dir = _persist_dir(embedding_name)
    if not Path(persist_dir).exists():
        raise FileNotFoundError(
            f"No vector store for '{embedding_name}' at {persist_dir}. "
            f"Run: python build_index.py --embedding {embedding_name}"
        )
    embeddings = get_embeddings(embedding_name)
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name=_collection_name(embedding_name),
    )
