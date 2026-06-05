"""
Retrieval strategies -- the heart of the capstone comparison.

Every strategy returns a standard LangChain Retriever so they can be swapped
freely in the RAG/CRAG pipelines. The full list lives in
config.STRATEGY_REGISTRY; the dispatcher below maps a name -> a retriever:

  1. "dense"           -> plain dense cosine similarity (Chroma).
  2. "hybrid"          -> dense + BM25 keyword, fused (EnsembleRetriever / RRF).
  3. "hybrid_rerank"   -> hybrid candidates re-scored by a cross-encoder reranker.
  4. "mmr"             -> Max-Marginal-Relevance over the dense store (diversity).
  5. "multi_query"     -> LLM expands the query into paraphrases, hybrid-fuses.
  6. "hyde"            -> embed an LLM-drafted hypothetical answer, then retrieve.
  7. "adaptive_hybrid" -> OUR OWN: query-shape-aware BM25<->dense weighting + MMR.
"""

import re
from typing import Callable, List

from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

import config
from src.embeddings import get_embeddings
from src.ingest import build_corpus
from src.vectorstore import load_vectorstore


class _FnRetriever(BaseRetriever):
    """Tiny adapter: turn a `query -> List[Document]` function into a Retriever.

    Used by strategies whose behaviour depends on the query string at call time
    (HyDE re-embeds the query; adaptive_hybrid re-weights per query), which a
    statically-built retriever object cannot express.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    fn: Callable[[str], List[Document]]

    def _get_relevant_documents(self, query, *, run_manager=None):  # noqa: D401
        return self.fn(query)


def _as_retriever(fn: Callable[[str], List[Document]]) -> BaseRetriever:
    return _FnRetriever(fn=fn)


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
    """Cross-encoder rerank (open-source, CPU). Same selection as LangChain's
    CrossEncoderReranker, but we score explicitly and stamp `relevance_score` into
    each surviving chunk's metadata so the UI can show the real number — not just a
    rank. (LangChain's compressor discards the score.)"""
    cross_encoder = HuggingFaceCrossEncoder(model_name=config.RERANKER_MODEL)

    def _retrieve(query: str) -> List[Document]:
        cands = base.invoke(query)
        if not cands:
            return []
        scores = cross_encoder.score([(query, d.page_content) for d in cands])
        ranked = sorted(zip(cands, scores), key=lambda x: float(x[1]), reverse=True)[:top_n]
        out = []
        for d, sc in ranked:
            d = Document(page_content=d.page_content, metadata={**d.metadata, "relevance_score": float(sc)})
            out.append(d)
        return out

    return _as_retriever(_retrieve)


def _mmr_retriever(embedding_name: str, k: int) -> BaseRetriever:
    """Max-Marginal-Relevance over the dense store: relevant but diverse hits."""
    store = load_vectorstore(embedding_name)
    return store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": max(4 * k, 20), "lambda_mult": 0.5},
    )


def _multi_query_retriever(embedding_name: str, corpus: List[Document], k: int) -> BaseRetriever:
    """LLM expands the question into several paraphrases; hybrid-retrieve each, fuse.

    The union across paraphrases can be large; we cap it (deduping by source+page)
    so the downstream CRAG grader doesn't have to grade dozens of chunks.
    """
    from src.rag import get_llm  # lazy: rag.py imports this module (avoid import cycle)

    base = _hybrid_retriever(embedding_name, corpus, k)
    mqr = MultiQueryRetriever.from_llm(retriever=base, llm=get_llm(), include_original=True)
    cap = max(2 * k, 8)

    def _retrieve(query: str) -> List[Document]:
        return _dedup_diverse(mqr.invoke(query), cap)

    return _as_retriever(_retrieve)


# HyDE: ask the LLM for a short hypothetical answer, embed THAT, and retrieve the
# real chunks nearest to it — helps when the question's wording is far from the
# paper's wording.
_HYDE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "You are writing a short, factual paragraph that would plausibly appear in "
                   "an AI/ML research paper and that directly answers the user's question. "
                   "Two or three sentences. Do not say you are unsure; just write the passage."),
        ("human", "{question}"),
    ]
)


def _hyde_retriever(embedding_name: str, k: int) -> BaseRetriever:
    from src.rag import get_llm  # lazy (import cycle)

    store = load_vectorstore(embedding_name)
    embedder = get_embeddings(embedding_name)
    drafter = _HYDE_PROMPT | get_llm() | StrOutputParser()

    def _retrieve(query: str) -> List[Document]:
        try:
            hypothetical = drafter.invoke({"question": query})
        except Exception:
            hypothetical = query  # fail-open: degrade to a normal dense query
        vec = embedder.embed_query(hypothetical or query)
        return store.similarity_search_by_vector(vec, k=k)

    return _as_retriever(_retrieve)


def _adaptive_weights(query: str):
    """OUR heuristic: read the query's shape to tilt BM25<->dense. Pure function.

    Keyword/acronym/short queries reward exact-match (BM25); long conceptual
    queries reward semantics (dense). Returns (dense_w, bm25_w, reason).
    """
    q = (query or "").strip()
    tokens = q.split()
    has_acronym = bool(re.search(r"\b[A-Z]{2,}\b", q))   # BERT, RAG, GPT, NSP
    has_quote = '"' in q or "'" in q
    has_number = bool(re.search(r"\d", q))
    short = len(tokens) <= 6
    keyword_signals = sum([has_acronym, has_quote, has_number, short])
    if keyword_signals >= 2:
        return 0.3, 0.7, (f"keyword-leaning (acronym={has_acronym}, quoted={has_quote}, "
                          f"number={has_number}, short={short}) → favour BM25 keyword match")
    if len(tokens) >= 14:
        return 0.7, 0.3, "long, conceptual query → favour dense semantic match"
    return 0.5, 0.5, "balanced query → equal dense/BM25 weighting"


def _dedup_diverse(docs: List[Document], k: int) -> List[Document]:
    """Lightweight MMR-style diversity: keep order but drop near-duplicate chunks
    from the same (source, page) so the context spans more of the corpus."""
    seen = set()
    out = []
    for d in docs:
        key = (d.metadata.get("source"), d.metadata.get("page_number"))
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
        if len(out) >= k:
            break
    return out


def _adaptive_hybrid_retriever(embedding_name: str, corpus: List[Document], k: int) -> BaseRetriever:
    """Build the ensemble with query-dependent weights at call time, then diversify."""
    pool = max(k, 8)

    def _retrieve(query: str) -> List[Document]:
        dense_w, bm25_w, _reason = _adaptive_weights(query)
        dense = _dense_retriever(embedding_name, pool)
        sparse = _bm25_retriever(corpus, pool)
        ensemble = EnsembleRetriever(retrievers=[dense, sparse], weights=[dense_w, bm25_w])
        return _dedup_diverse(ensemble.invoke(query), k)

    return _as_retriever(_retrieve)


def get_retriever(
    strategy: str = config.DEFAULT_STRATEGY,
    embedding_name: str = config.DEFAULT_EMBEDDING,
    k: int = config.TOP_K,
) -> BaseRetriever:
    """Build a retriever for the requested strategy (see config.STRATEGY_REGISTRY)."""
    # Strategies that don't need the in-memory BM25 corpus.
    if strategy == "dense":
        return _dense_retriever(embedding_name, k)
    if strategy == "mmr":
        return _mmr_retriever(embedding_name, k)
    if strategy == "hyde":
        return _hyde_retriever(embedding_name, k)

    # BM25-backed strategies need the in-memory corpus.
    corpus = build_corpus()

    if strategy == "hybrid":
        return _hybrid_retriever(embedding_name, corpus, k)

    if strategy == "hybrid_rerank":
        # Fetch a wider candidate pool, then rerank down to RERANK_TOP_N.
        hybrid = _hybrid_retriever(embedding_name, corpus, k=max(k, 8))
        return _rerank_retriever(hybrid, top_n=config.RERANK_TOP_N)

    if strategy == "multi_query":
        return _multi_query_retriever(embedding_name, corpus, k)

    if strategy == "adaptive_hybrid":
        return _adaptive_hybrid_retriever(embedding_name, corpus, k)

    raise ValueError(
        f"Unknown strategy '{strategy}'. Valid: {', '.join(config.STRATEGY_NAMES)}"
    )
