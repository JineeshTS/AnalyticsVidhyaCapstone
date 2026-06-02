"""
Retrieval strategies -- the heart of the capstone comparison.

Three strategies, all returning a standard LangChain Retriever so they can be
swapped freely in the RAG/CRAG pipelines:

  1. "dense"          -> plain dense cosine similarity (Chroma).
  2. "hybrid"         -> dense + BM25 keyword, fused (EnsembleRetriever / RRF).
  3. "hybrid_rerank"  -> hybrid candidates re-scored by a cross-encoder reranker.
"""

from typing import List

from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

import config
from src.ingest import build_corpus
from src.vectorstore import load_vectorstore


def _dense_retriever(embedding_name: str, k: int) -> BaseRetriever:
    store = load_vectorstore(embedding_name)
    return store.as_retriever(search_kwargs={"k": k})


def _bm25_retriever(corpus: List[Document], k: int) -> BaseRetriever:
    retriever = BM25Retriever.from_documents(corpus)
    retriever.k = k
    return retriever


def _hybrid_retriever(embedding_name: str, corpus: List[Document], k: int) -> BaseRetriever:
    """Dense + BM25 fused with Reciprocal Rank Fusion (equal weights)."""
    dense = _dense_retriever(embedding_name, k)
    sparse = _bm25_retriever(corpus, k)
    return EnsembleRetriever(retrievers=[dense, sparse], weights=[0.5, 0.5])


def _rerank_retriever(base: BaseRetriever, top_n: int) -> BaseRetriever:
    """Wrap a base retriever with a cross-encoder reranker (open-source, CPU)."""
    cross_encoder = HuggingFaceCrossEncoder(model_name=config.RERANKER_MODEL)
    compressor = CrossEncoderReranker(model=cross_encoder, top_n=top_n)
    return ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base
    )


def get_retriever(
    strategy: str = config.DEFAULT_STRATEGY,
    embedding_name: str = config.DEFAULT_EMBEDDING,
    k: int = config.TOP_K,
) -> BaseRetriever:
    """
    Build a retriever for the requested strategy.

    strategy: "dense" | "hybrid" | "hybrid_rerank"
    """
    if strategy == "dense":
        return _dense_retriever(embedding_name, k)

    # Both hybrid strategies need the in-memory corpus for BM25.
    corpus = build_corpus()

    if strategy == "hybrid":
        return _hybrid_retriever(embedding_name, corpus, k)

    if strategy == "hybrid_rerank":
        # Fetch a wider candidate pool, then rerank down to RERANK_TOP_N.
        hybrid = _hybrid_retriever(embedding_name, corpus, k=max(k, 8))
        return _rerank_retriever(hybrid, top_n=config.RERANK_TOP_N)

    raise ValueError(
        f"Unknown strategy '{strategy}'. Use: dense | hybrid | hybrid_rerank"
    )
