"""
Agentic Corrective RAG (CRAG) -- stretch goal.

A LangGraph state machine that makes retrieval self-correcting:

    retrieve -> grade each doc for relevance
        |-- all relevant            -> generate
        |-- some/none relevant      -> rewrite query -> web search -> generate

This means when the indexed papers don't cover a question well, the bot falls
back to a live web search instead of confidently answering from weak context.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, TypedDict

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

import config
from src.rag import ANSWER_PROMPT, format_context, format_sources, get_llm
from src.retrievers import get_retriever
from src.websearch import web_search


# --- Per-(strategy, embedding) retriever cache -----------------------------
# The graph is compiled once, but the strategy is now chosen PER QUERY (the gate
# can auto-route). So the retrieve node builds/looks up the retriever at call
# time. Building a hybrid/BM25 retriever re-reads the corpus, so we cache them.
# Cleared by webapp's _rebuild_state() whenever the corpus changes (uploads), so
# BM25-backed retrievers never serve stale chunks.
_RETRIEVER_CACHE: dict = {}
_RETRIEVER_CACHE_LOCK = threading.Lock()


def _get_cached_retriever(strategy: str, embedding_name: str) -> BaseRetriever:
    key = (strategy, embedding_name)
    r = _RETRIEVER_CACHE.get(key)
    if r is None:
        with _RETRIEVER_CACHE_LOCK:
            r = _RETRIEVER_CACHE.get(key)  # double-checked
            if r is None:
                r = get_retriever(strategy, embedding_name)
                _RETRIEVER_CACHE[key] = r
    return r


def clear_retriever_cache() -> None:
    """Drop all cached retrievers (call after the corpus changes)."""
    with _RETRIEVER_CACHE_LOCK:
        _RETRIEVER_CACHE.clear()


def get_gate_llm():
    """Fast classifier model for the query gate (cheaper than the generator)."""
    return get_llm(config.GATE_MODEL)


# --- Document relevance grader (structured output) -------------------------
class GradeDocuments(BaseModel):
    """Binary relevance score for a retrieved document."""

    binary_score: str = Field(description="'yes' if relevant to the question, else 'no'")


GRADE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You grade whether a retrieved document actually helps ANSWER a "
            "user's question. Grade 'yes' only if the document contains "
            "information about the specific topic, method, or entity the "
            "question asks about. Mere keyword overlap is NOT enough: a document "
            "that shares a few words but is about a different topic must be "
            "graded 'no'. If the document does not contain information that "
            "would help answer the question, grade 'no'.",
        ),
        ("human", "Document:\n{document}\n\nQuestion: {question}"),
    ]
)

# --- Query rewriter for the web-search fallback ----------------------------
REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the user's question into a concise, keyword-rich web search "
            "query that would find authoritative information on the topic. "
            "Return only the rewritten query.",
        ),
        ("human", "{question}"),
    ]
)


# --- Pre-RAG query gate (classify + route + clarify) -----------------------
# NOTE: the Claude-CLI structured-output adapter forwards only each field's
# *description* to the model (not Literal/enum types), so the allowed values are
# spelled out inside the descriptions, and the node post-validates them.
class QueryGate(BaseModel):
    """Pre-retrieval classification of the user's question."""

    quality: str = Field(description="Exactly one of: clear, ambiguous, too_broad, out_of_scope, chitchat")
    query_type: str = Field(description="Exactly one of: conceptual, keyword, comparison, definition, factoid")
    recommended_strategy: str = Field(description=(
        "Best retrieval strategy, exactly one of: dense, hybrid, hybrid_rerank, "
        "mmr, multi_query, hyde, adaptive_hybrid"))
    clarifying_questions: list = Field(description=(
        "If quality is ambiguous or too_broad, 1-3 short questions to ask the user "
        "before searching; otherwise an empty list []"))
    reasoning: str = Field(description="One short sentence explaining the classification and routing")


GATE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a query router for a RAG system over landmark AI/ML research "
            "papers (Transformer/Attention, BERT, GPT-3, RAG, Chain-of-Thought, "
            "InstructGPT). Classify the user's question.\n"
            "- quality is 'ambiguous' or 'too_broad' ONLY when retrieval would likely "
            "fail without clarification (e.g. 'tell me about models', 'explain it'). "
            "Otherwise quality='clear' and clarifying_questions=[].\n"
            "- 'chitchat' for greetings/meta; 'out_of_scope' if clearly unrelated to AI/ML.\n"
            "- Pick recommended_strategy by query shape: short / acronym / exact-keyword "
            "→ 'adaptive_hybrid' or 'hybrid'; long / conceptual → 'hybrid_rerank' or "
            "'multi_query'; vague-wording that needs paraphrase → 'hyde'; default "
            "'hybrid_rerank' when unsure.\n"
            "Do NOT answer the question itself.",
        ),
        ("human", "Question: {question}"),
    ]
)


class CRAGState(TypedDict):
    question: str
    documents: List[Document]
    generation: str
    used_web_search: bool
    trace: list                   # step-by-step processing log (for transparency)
    # --- query gate / routing ---
    strategy: str                 # requested strategy ("auto" or a registry name)
    embedding: str                # embedding to retrieve with
    gate: dict                    # raw gate classification
    recommended_strategy: str     # resolved strategy actually used
    clarifying_questions: list    # set only when the gate short-circuits
    routed: bool                  # True when "auto" changed the strategy
    short_circuit: bool           # True when the gate ends early (ask, don't answer)


def _grader():
    return GRADE_PROMPT | get_llm(config.GRADER_MODEL).with_structured_output(GradeDocuments)


def _label(d: Document) -> dict:
    """Compact, presentation-friendly reference to a chunk (for the trace)."""
    return {
        "title": config.display_title(d.metadata.get("source", ""), d.metadata.get("title", "Unknown")),
        "page": d.metadata.get("page_number", "?"),
        "web": d.metadata.get("origin") == "web_search",
    }


# --- Graph nodes -----------------------------------------------------------
def _make_nodes():
    grader = _grader()
    rewriter = REWRITE_PROMPT | get_llm() | StrOutputParser()
    generator = ANSWER_PROMPT | get_llm() | StrOutputParser()
    gate_chain = GATE_PROMPT | get_gate_llm().with_structured_output(QueryGate)

    def gate(state: CRAGState) -> CRAGState:
        """Pre-RAG: classify the question, route the strategy, or ask for clarification."""
        t = time.perf_counter()
        requested = state.get("strategy") or config.DEFAULT_STRATEGY
        try:
            g = gate_chain.invoke({"question": state["question"]})
            data = g.model_dump() if hasattr(g, "model_dump") else dict(g)
            quality = (data.get("quality") or "clear").strip().lower()
            qtype = (data.get("query_type") or "").strip().lower()
            rec = (data.get("recommended_strategy") or "").strip()
            if rec not in config.STRATEGY_NAMES:          # coerce hallucinated names
                rec = config.DEFAULT_STRATEGY
            clar = data.get("clarifying_questions") or []
            clar = [str(c) for c in clar][:3] if isinstance(clar, list) else []
            reasoning = data.get("reasoning", "")
        except Exception as e:  # FAIL-OPEN: behave as if the gate never ran
            quality, qtype, reasoning = "clear", "", f"gate skipped ({type(e).__name__})"
            rec = requested if requested != "auto" else config.DEFAULT_STRATEGY
            clar = []

        resolved = rec if requested == "auto" else requested
        routed = requested == "auto"
        short = quality in ("ambiguous", "too_broad") and bool(clar)
        if short:
            detail = f"Classified as {quality}; asking {len(clar)} clarifying question(s) before searching."
        else:
            rt = f"routed to '{resolved}'" if routed else f"using '{resolved}'"
            detail = f"Classified as {quality}{('/' + qtype) if qtype else ''}; {rt}."
        ev = {"step": "gate", "label": "0 · Analyze query", "detail": detail,
              "quality": quality, "query_type": qtype, "recommended_strategy": rec,
              "resolved_strategy": resolved, "requested_strategy": requested, "routed": routed,
              "reasoning": reasoning, "clarifying_questions": clar if short else [],
              "ms": round((time.perf_counter() - t) * 1000)}
        return {**state,
                "gate": {"quality": quality, "query_type": qtype,
                         "recommended_strategy": rec, "reasoning": reasoning},
                "strategy": resolved, "recommended_strategy": resolved,
                "clarifying_questions": clar if short else [],
                "routed": routed, "short_circuit": short,
                "trace": state.get("trace", []) + [ev]}

    def retrieve(state: CRAGState) -> CRAGState:
        t = time.perf_counter()
        strat = state.get("strategy") or config.DEFAULT_STRATEGY
        emb = state.get("embedding") or config.DEFAULT_EMBEDDING
        retriever = _get_cached_retriever(strat, emb)
        docs = retriever.invoke(state["question"])
        label = config.STRATEGY_REGISTRY.get(strat, {}).get("label", strat)
        ev = {"step": "retrieve", "label": "1 · Retrieve",
              "detail": f"Retrieved {len(docs)} candidate chunks using the '{label}' strategy on the '{emb}' embedding.",
              "strategy": strat, "embedding": emb,
              "count": len(docs), "ms": round((time.perf_counter() - t) * 1000)}
        return {**state, "documents": docs, "used_web_search": False,
                "trace": state.get("trace", []) + [ev]}

    def grade_documents(state: CRAGState) -> CRAGState:
        t = time.perf_counter()
        docs = state["documents"]
        question = state["question"]

        def grade_one(d: Document) -> bool:
            try:
                score = grader.invoke({"document": d.page_content, "question": question})
                return score.binary_score.strip().lower() == "yes"
            except Exception:
                return True  # fail-open: keep the chunk if the grader errors

        # Grade all chunks concurrently — each grade is one LLM call.
        if docs:
            workers = min(len(docs), config.GRADE_CONCURRENCY)
            with ThreadPoolExecutor(max_workers=workers) as ex:
                verdicts = list(ex.map(grade_one, docs))
        else:
            verdicts = []
        kept = [d for d, v in zip(docs, verdicts) if v]
        dropped = [d for d, v in zip(docs, verdicts) if not v]

        decision = ("At least one chunk is relevant → answer from the papers."
                    if kept else "No chunk is relevant → fall back to a live web search.")
        evs = [
            {"step": "grade", "label": "2 · Grade relevance",
             "detail": f"An LLM graded all {len(docs)} chunks in parallel; kept {len(kept)} as actually relevant.",
             "kept": [_label(d) for d in kept], "dropped": [_label(d) for d in dropped],
             "ms": round((time.perf_counter() - t) * 1000)},
            {"step": "decide", "label": "3 · Decision", "detail": decision},
        ]
        return {**state, "documents": kept, "trace": state.get("trace", []) + evs}

    def transform_query(state: CRAGState) -> CRAGState:
        t = time.perf_counter()
        better = rewriter.invoke({"question": state["question"]})
        ev = {"step": "rewrite", "label": "4 · Rewrite query",
              "detail": "Rewrote the question into a web-search query.",
              "from": state["question"], "to": better, "ms": round((time.perf_counter() - t) * 1000)}
        return {**state, "question": better, "trace": state.get("trace", []) + [ev]}

    def do_web_search(state: CRAGState) -> CRAGState:
        t = time.perf_counter()
        web_docs = web_search(state["question"])
        providers = sorted({d.metadata.get("provider", "ddg") for d in web_docs}) or ["ddg"]
        provider_label = {"ddg": "DuckDuckGo", "brave": "Brave", "serper": "Serper"}
        pretty = ", ".join(provider_label.get(p, p) for p in providers)
        ev = {"step": "websearch", "label": "5 · Web search",
              "detail": f"Fetched {len(web_docs)} live web results ({pretty}).",
              "provider": pretty, "providers": providers,
              "count": len(web_docs), "ms": round((time.perf_counter() - t) * 1000)}
        return {**state, "documents": state["documents"] + web_docs,
                "used_web_search": True, "trace": state.get("trace", []) + [ev]}

    def generate(state: CRAGState) -> CRAGState:
        t = time.perf_counter()
        context = format_context(state["documents"])
        answer = generator.invoke({"context": context, "question": state["question"]})
        ev = {"step": "generate", "label": "6 · Generate",
              "detail": f"Claude ({config.CLAUDE_MODEL}) wrote the answer from {len(state['documents'])} context chunks.",
              "context": [_label(d) for d in state["documents"]],
              "context_text": context[:6000], "model": config.CLAUDE_MODEL,
              "ms": round((time.perf_counter() - t) * 1000)}
        return {**state, "generation": answer, "trace": state.get("trace", []) + [ev]}

    return gate, retrieve, grade_documents, transform_query, do_web_search, generate


def _decide_after_gate(state: CRAGState) -> str:
    """Ask for clarification (end early) or proceed to retrieval."""
    return "clarify" if state.get("short_circuit") else "retrieve"


def _decide_after_grading(state: CRAGState) -> str:
    """If no relevant paper chunks survived, fall back to web search."""
    return "generate" if state["documents"] else "transform_query"


def build_crag_app(
    strategy: str = config.DEFAULT_STRATEGY,
    embedding_name: str = config.DEFAULT_EMBEDDING,
    retriever: BaseRetriever = None,
):
    """Compile and return the CRAG LangGraph app.

    The graph is strategy-AGNOSTIC at compile time: the retrieve node selects the
    retriever per query from state["strategy"]/["embedding"] (so the query gate
    can auto-route). `strategy`/`embedding_name` only seed the per-run default;
    `retriever` is accepted for backwards compatibility but no longer used.
    """
    gate, retrieve, grade_documents, transform_query, do_web_search, generate = _make_nodes()

    g = StateGraph(CRAGState)
    g.add_node("gate", gate)
    g.add_node("retrieve", retrieve)
    g.add_node("grade_documents", grade_documents)
    g.add_node("transform_query", transform_query)
    g.add_node("web_search", do_web_search)
    g.add_node("generate", generate)

    g.add_edge(START, "gate")
    g.add_conditional_edges(
        "gate", _decide_after_gate,
        {"clarify": END, "retrieve": "retrieve"},
    )
    g.add_edge("retrieve", "grade_documents")
    g.add_conditional_edges(
        "grade_documents",
        _decide_after_grading,
        {"generate": "generate", "transform_query": "transform_query"},
    )
    g.add_edge("transform_query", "web_search")
    g.add_edge("web_search", "generate")
    g.add_edge("generate", END)

    return g.compile()


def _initial_state(question: str, strategy: str, embedding_name: str) -> dict:
    return {"question": question, "documents": [], "generation": "",
            "used_web_search": False, "trace": [],
            "strategy": strategy, "embedding": embedding_name,
            "gate": {}, "recommended_strategy": strategy,
            "clarifying_questions": [], "routed": False, "short_circuit": False}


def answer_question_crag(question: str, app=None,
                         strategy: str = config.DEFAULT_STRATEGY,
                         embedding_name: str = config.DEFAULT_EMBEDDING, **kwargs) -> dict:
    """Run the CRAG graph and return answer + sources in the same shape as rag.py
    (plus gate/routing fields, and clarifying_questions when the gate short-circuits)."""
    if app is None:
        app = build_crag_app(strategy=strategy, embedding_name=embedding_name, **kwargs)
    final = app.invoke(_initial_state(question, strategy, embedding_name))
    return {
        "answer": final.get("generation", ""),
        "sources": format_sources(final["documents"]),
        "documents": final["documents"],
        "used_web_search": final["used_web_search"],
        "trace": final.get("trace", []),
        "gate": final.get("gate", {}),
        "routed_strategy": final.get("recommended_strategy"),
        "routed": bool(final.get("routed")),
        "clarifying_questions": final.get("clarifying_questions", []),
        "needs_clarification": bool(final.get("short_circuit")),
    }


if __name__ == "__main__":
    import sys

    q = sys.argv[1] if len(sys.argv) > 1 else "What is retrieval augmented generation?"
    res = answer_question_crag(q)
    print("ANSWER:\n", res["answer"])
    print("\nUsed web search:", res["used_web_search"])
    for s in res["sources"]:
        print(f"  - {s['title']} (page {s['page']})")
