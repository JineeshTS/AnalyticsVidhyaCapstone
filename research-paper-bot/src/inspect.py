"""
Retrieval internals — expose each stage of the pipeline for visualization.

The chat answer hides *how* context is found. This module returns the
intermediate results of every retrieval stage so the web UI can show, for a
given query:

  1. dense    — top-k by embedding cosine similarity (with relevance scores)
  2. bm25     — top-k by BM25 keyword scoring (with BM25 scores)
  3. hybrid   — the two fused with Reciprocal Rank Fusion (final hybrid order)
  4. reranked — hybrid candidates re-scored by the cross-encoder, reordered

This is what makes "how the reranker works" visible: you can see a chunk move
up or down between the hybrid and reranked columns based on its cross-encoder
relevance score.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document

import config
from src.vectorstore import load_vectorstore


def _item(doc: Document, rank: int, score: Optional[float]) -> Dict:
    return {
        "rank": rank,
        "title": config.display_title(doc.metadata.get("source", ""), doc.metadata.get("title", "Unknown")),
        "page": doc.metadata.get("page_number", "?"),
        "source": doc.metadata.get("source", ""),
        "score": None if score is None else round(float(score), 4),
        "snippet": doc.page_content[:240].strip().replace("\n", " "),
    }


def _key(doc: Document) -> tuple:
    return (doc.metadata.get("source", ""), doc.metadata.get("page_number", "?"),
            doc.page_content[:60])


def inspect_query(
    query: str,
    corpus: List[Document],
    embedding_name: str = None,
    k: int = 8,
    rerank_top_n: int = None,
    cross_encoder=None,
) -> Dict:
    """Run every retrieval stage for `query` and return their ranked results."""
    embedding_name = embedding_name or config.DEFAULT_EMBEDDING
    rerank_top_n = rerank_top_n or config.RERANK_TOP_N

    # --- 1. dense (embedding cosine) ---
    store = load_vectorstore(embedding_name)
    dense_pairs = store.similarity_search_with_relevance_scores(query, k=k)
    dense = [_item(d, i + 1, s) for i, (d, s) in enumerate(dense_pairs)]

    # --- 2. bm25 (keyword) — pull scores out of the underlying vectorizer ---
    bm25 = BM25Retriever.from_documents(corpus)
    bm25.k = k
    tokens = query.lower().split()
    try:
        raw_scores = bm25.vectorizer.get_scores(tokens)  # aligned to bm25.docs
        ranked = sorted(zip(bm25.docs, raw_scores), key=lambda x: x[1], reverse=True)[:k]
        bm25_items = [_item(d, i + 1, s) for i, (d, s) in enumerate(ranked)]
    except Exception:
        # Fallback: ranked docs without numeric scores.
        docs = bm25.invoke(query)[:k]
        bm25_items = [_item(d, i + 1, None) for i, d in enumerate(docs)]

    # --- 3. hybrid (RRF fusion of dense + bm25) ---
    dense_ret = store.as_retriever(search_kwargs={"k": k})
    hybrid_ret = EnsembleRetriever(retrievers=[dense_ret, bm25], weights=[0.5, 0.5])
    hybrid_docs = hybrid_ret.invoke(query)
    hybrid = [_item(d, i + 1, None) for i, d in enumerate(hybrid_docs)]

    # --- 4. reranked (cross-encoder over the hybrid candidates) ---
    if cross_encoder is None:
        from langchain_community.cross_encoders import HuggingFaceCrossEncoder
        cross_encoder = HuggingFaceCrossEncoder(model_name=config.RERANKER_MODEL)
    pairs = [(query, d.page_content) for d in hybrid_docs]
    ce_scores = cross_encoder.score(pairs) if pairs else []
    reranked_sorted = sorted(zip(hybrid_docs, ce_scores), key=lambda x: x[1], reverse=True)
    reranked = [_item(d, i + 1, s) for i, (d, s) in enumerate(reranked_sorted)]
    kept = {_key(d) for d, _ in reranked_sorted[:rerank_top_n]}
    for it, (d, _) in zip(reranked, reranked_sorted):
        it["kept"] = _key(d) in kept  # marks which survive into the final top-N

    return {
        "query": query,
        "embedding": embedding_name,
        "rerank_top_n": rerank_top_n,
        "stages": {
            "dense": dense,
            "bm25": bm25_items,
            "hybrid": hybrid,
            "reranked": reranked,
        },
    }
