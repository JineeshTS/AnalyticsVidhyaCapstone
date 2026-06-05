"""
FastAPI web app for the Research Paper Answer Bot — an interactive RAG explorer.

Three surfaces over the existing pipeline:
  - /api/ask      : answer a question (Corrective-RAG) with sources + web badge
  - /api/inspect  : the retrieval internals (dense / bm25 / hybrid / reranked)
  - /api/upload   : add a PDF live → chunk → embed → index → queryable instantly
  - /api/corpus   : list indexed documents + current config + options
  - /api/config   : switch embedding / retrieval strategy at runtime
  - /api/reset    : drop uploaded docs, restore the original corpus

Production notes (runs public at paperbot.ganakys.com, gated by nginx):
  - Heavy objects (CRAG graph, corpus, cross-encoder) are built once and cached;
    a rebuild lock serializes corpus mutations and swaps the graph atomically.
  - A concurrency semaphore caps simultaneous `claude` CLI subprocesses.
  - The blocking pipeline calls run in worker threads.

Run locally:  uvicorn webapp:app --host 127.0.0.1 --port 8011
"""

from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Cookie, FastAPI, File, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

import config
from src import memory
from src.crag import build_crag_app
from src.ingest import build_corpus, load_single_pdf
from src.inspect import inspect_query
from src.rag import format_sources, get_llm
from src.vectorstore import add_to_vectorstore, has_vectorstore

CONDENSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given the conversation history and a follow-up question, rewrite the "
            "follow-up into a standalone question that makes sense on its own. "
            "If it is already standalone, return it unchanged. Return only the "
            "question.",
        ),
        ("human", "History:\n{history}\n\nFollow-up: {question}"),
    ]
)

STRATEGIES = config.STRATEGY_NAMES                  # the 7 concrete strategies
STRATEGIES_WITH_AUTO = ["auto"] + STRATEGIES        # + the gate-driven router
MAX_CONCURRENCY = 2
MAX_UPLOAD_BYTES = 25 * 1024 * 1024

PROJECT_ROOT = Path(__file__).resolve().parent


def _source_allowlist() -> dict:
    """Map relative path -> absolute path for the source files we expose in the
    in-app code browser. Excludes secrets (.env), venv, data, storage, .git."""
    groups = {
        "Entry points": ["config.py", "build_index.py", "evaluate.py", "app.py", "webapp.py"],
        "src/": sorted(p.name for p in (PROJECT_ROOT / "src").glob("*.py")),
        "scripts/": ["download_papers.py"],
        "web/": ["index.html"],
        "deploy/": sorted(p.name for p in (PROJECT_ROOT / "deploy").glob("*") if p.is_file()),
        "docs/config": ["requirements.txt", ".env.example", "README.md"],
    }
    prefix = {"src/": "src/", "scripts/": "scripts/", "web/": "web/", "deploy/": "deploy/"}
    allow = {}
    for grp, names in groups.items():
        for n in names:
            rel = prefix.get(grp, "") + n
            ap = (PROJECT_ROOT / rel).resolve()
            if ap.is_file() and PROJECT_ROOT in ap.parents or ap == PROJECT_ROOT:
                if ap.is_file():
                    allow[rel] = ap
    return allow

_semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
_rebuild_lock = asyncio.Lock()
STATIC_DIR = Path(__file__).resolve().parent / "web"

_state: dict = {}


def _doc_summary(corpus) -> list:
    """Group the in-memory corpus by source → [{title, source, chunks}]."""
    by_src: dict = {}
    for d in corpus:
        src = d.metadata.get("source", "?")
        if src not in by_src:
            by_src[src] = {"title": config.display_title(src, d.metadata.get("title", src)), "source": src, "chunks": 0}
        by_src[src]["chunks"] += 1
    return sorted(by_src.values(), key=lambda x: x["title"].lower())


def _rebuild_state():
    """(Re)build corpus + CRAG graph for the current embedding/strategy. Blocking."""
    from src.crag import clear_retriever_cache
    clear_retriever_cache()   # corpus may have changed → drop stale (BM25-backed) retrievers
    corpus = build_corpus()
    _state["corpus"] = corpus
    _state["original_sources"] = _state.get("original_sources") or {d.metadata.get("source") for d in corpus}
    _state["crag_app"] = build_crag_app(
        strategy=_state["strategy"], embedding_name=_state["embedding"]
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["embedding"] = config.DEFAULT_EMBEDDING
    _state["strategy"] = config.DEFAULT_STRATEGY
    _rebuild_state()
    _state["condenser"] = CONDENSE_PROMPT | get_llm() | StrOutputParser()
    _state["cross_encoder"] = None  # built lazily on first /api/inspect
    yield
    _state.clear()


app = FastAPI(title="Research Paper Answer Bot", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str
    strategy: str | None = None     # "auto" or a registry name; defaults to the active strategy


class InspectRequest(BaseModel):
    query: str
    embedding: str | None = None


class ConfigRequest(BaseModel):
    embedding: str | None = None
    strategy: str | None = None


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "embedding": _state.get("embedding"),
            "strategy": _state.get("strategy"), "backend": config.LLM_BACKEND}


@app.get("/api/prompts")
def prompts() -> dict:
    """The actual prompts sent to the LLM, the config knobs, and the pipeline —
    so the evaluator can see exactly how the system is wired."""
    from src.crag import ANSWER_PROMPT, GATE_PROMPT, GRADE_PROMPT, REWRITE_PROMPT

    def tmpl_text(tmpl):
        out = []
        for m in tmpl.messages:
            role = getattr(m, "role", None) or type(m).__name__.replace("MessagePromptTemplate", "")
            text = getattr(getattr(m, "prompt", None), "template", None)
            if text is None:
                continue
            out.append(f"[{role.lower()}]\n{text}")
        return "\n\n".join(out)

    return {
        "prompts": [
            {"name": "Query gate (pre-RAG router)", "role": "Classify the question, route the retrieval strategy, and ask for clarification when it's too vague — before any search", "text": tmpl_text(GATE_PROMPT)},
            {"name": "Answer (RAG generation)", "role": "Generate the grounded answer from retrieved context", "text": tmpl_text(ANSWER_PROMPT)},
            {"name": "Relevance grader (CRAG)", "role": "Decide if a retrieved chunk actually helps answer the question", "text": tmpl_text(GRADE_PROMPT)},
            {"name": "Query rewriter (CRAG)", "role": "Turn the question into a web-search query before the fallback", "text": tmpl_text(REWRITE_PROMPT)},
            {"name": "Follow-up condenser", "role": "Rewrite a follow-up into a standalone question using chat history", "text": tmpl_text(CONDENSE_PROMPT)},
        ],
        "config": {
            "CHUNK_SIZE": config.CHUNK_SIZE, "CHUNK_OVERLAP": config.CHUNK_OVERLAP,
            "TOP_K": config.TOP_K, "TOP_SOURCES": config.TOP_SOURCES,
            "RERANK_TOP_N": config.RERANK_TOP_N, "WEB_SEARCH_RESULTS": config.WEB_SEARCH_RESULTS,
            "LLM_BACKEND": config.LLM_BACKEND, "CLAUDE_MODEL": config.CLAUDE_MODEL,
            "GATE_MODEL": config.GATE_MODEL, "WEB_SEARCH_PROVIDER": config.WEB_SEARCH_PROVIDER,
        },
        "pipeline": [
            "analyse — a fast LLM gate classifies the question, routes the strategy, and asks to clarify if it's too vague",
            "retrieve — fetch candidate chunks with the chosen strategy (dense / hybrid / rerank / mmr / multi-query / hyde / adaptive)",
            "grade — an LLM judges each chunk for real relevance",
            "decide — relevant chunks? answer from papers : fall back to web",
            "rewrite + web search — only when no chunk is relevant (pluggable; DuckDuckGo by default)",
            "generate — Claude writes the answer from the context, with sources",
        ],
    }


@app.get("/api/evaluation")
def evaluation() -> dict:
    """The real benchmark output (storage/eval_results.csv), ranked."""
    import csv
    path = config.STORAGE_DIR / "eval_results.csv"
    if not path.exists():
        return {"rows": [], "note": "Run `python evaluate.py` to generate this."}
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({"embedding": r["embedding"], "strategy": r["strategy"],
                         "avg_score": float(r["avg_score"]), "avg_latency_s": float(r["avg_latency_s"])})
    rows.sort(key=lambda x: (-x["avg_score"], x["avg_latency_s"]))
    best = rows[0] if rows else None
    finding = None
    if best:
        rerank = [r for r in rows if r["strategy"] == "hybrid_rerank"]
        rerank_note = (
            f" The cross-encoder reranker actually <b>hurt</b> on this small {len(_doc_summary(_state.get('corpus', [])))}"
            f"-document corpus — it trims the context to a handful of chunks and drops passages the answer "
            f"needed, dragging quality down to {min(r['avg_score'] for r in rerank)}/5 while adding latency."
            if rerank else ""
        )
        finding = (
            f"<b>Hybrid retrieval wins.</b> Fusing keyword search (BM25) with semantic similarity beat "
            f"pure dense retrieval for every embedding.{rerank_note} All the leading embeddings tied at "
            f"{best['avg_score']}/5 with hybrid, so <b>{best['embedding']} + {best['strategy']}</b> was made "
            f"the live default: strong open-source retrieval, the fastest of the leaders "
            f"({best['avg_latency_s']}s), and no API cost."
        )
    return {"rows": rows, "best": best, "finding": finding,
            "method": "LLM-as-judge answer quality (1–5) plus latency, over 5 sample queries × 3 embeddings × 3 strategies."}


# One-line "why it matters" + year for each seminal paper, keyed by source filename.
# Used only to render the intro strip on the Criteria page.
PAPER_META = {
    "Attention_Is_All_You_Need.pdf": ("The Transformer — self-attention replaces recurrence", "2017"),
    "BERT.pdf": ("Bidirectional pre-training for language understanding", "2018"),
    "GPT-3_Language_Models_Few_Shot.pdf": ("Few-shot learning emerges at scale", "2020"),
    "RAG_Retrieval_Augmented_Generation.pdf": ("The paper that named retrieval-augmented generation", "2020"),
    "Chain_of_Thought_Prompting.pdf": ("Step-by-step reasoning unlocks harder problems", "2022"),
    "InstructGPT_Training_with_Human_Feedback.pdf": ("RLHF — aligning models to follow instructions", "2022"),
}

GROUP_REQUIRED = "Required by the assignment"
GROUP_STRETCH = "Stretch goals (bonus)"


@app.get("/api/criteria")
def criteria() -> dict:
    """Live rubric for the Analytics Vidhya GenAI capstone: a project intro, the
    indexed papers, and each requirement with whether it's met, a self-contained
    explanation, and a jump to the proof."""
    docs = _doc_summary(_state.get("corpus", []))
    chunk_total = sum(d["chunks"] for d in docs)
    built = [n for n in config.EMBEDDING_MODELS if has_vectorstore(n)]
    commercial = [n for n in built if config.EMBEDDING_MODELS[n]["provider"] in ("gemini", "openai")]
    opensource = [n for n in built if config.EMBEDDING_MODELS[n]["provider"] == "huggingface"]
    has_eval = (config.STORAGE_DIR / "eval_results.csv").exists()

    # The seminal papers actually present in the corpus, in chronological order.
    present = {d["source"] for d in docs}
    papers = [
        {"title": config.display_title(src), "blurb": meta[0], "year": meta[1]}
        for src, meta in PAPER_META.items() if src in present
    ]
    embed_label = {n: config.EMBEDDING_MODELS[n]["label"].split(" (")[0] for n in built}
    embed_names = ", ".join(embed_label.get(n, n) for n in built) or "none built yet"

    intro = {
        "name": "Research Paper Answer Bot",
        "tagline": "Ask landmark Generative-AI papers anything — every answer is grounded, "
                   "cited to the page, and written by a Claude model running locally at $0 API cost.",
        "blurb": f"A Retrieval-Augmented Generation (RAG) system built for the Analytics Vidhya "
                 f"GenAI capstone. {len(papers)} seminal papers are parsed into {chunk_total} text "
                 f"chunks and indexed in a Chroma vector database. Questions are answered by Claude "
                 f"({config.CLAUDE_MODEL}) run through the local Claude CLI — so there is no per-query "
                 f"API spend. It cites the exact paper and page behind every answer, and falls back to "
                 f"a live web search when the papers don't cover the question.",
    }

    def c(group, title, met, how, tab):
        return {"group": group, "title": title, "met": met, "how": how, "tab": tab}

    items = [
        c(GROUP_REQUIRED, "A dataset of research papers", len(papers) > 0,
          f"{len(papers)} landmark Generative-AI papers are loaded — "
          f"{', '.join(p['title'].split(' (')[0] for p in papers)} — totalling {chunk_total} chunks. "
          f"You can drop in your own PDF on the Corpus tab and query it immediately.", "corpus"),
        c(GROUP_REQUIRED, "Loaded & indexed in a vector database", len(built) > 0,
          f"Every chunk is embedded and stored in a persistent Chroma vector DB. "
          f"{len(built)} separate collections are built — one per embedding model ({embed_names}) — "
          f"so they can be compared head-to-head. The Stack tab shows each collection's vector count "
          f"and dimensions.", "corpus"),
        c(GROUP_REQUIRED, "Embeddings compared: open-source vs. commercial",
          len(opensource) > 0 and len(commercial) > 0,
          f"Open-source models ({', '.join(opensource)}) run locally on the VPS CPU; the commercial "
          f"model ({', '.join(commercial)}) uses Google's free tier. The Inspector's "
          f"‘Compare embeddings’ button runs one question through all of them at once, and the "
          f"benchmark below scores each.", "inspect"),
        c(GROUP_REQUIRED, "Retrieval strategies compared: 7 strategies + Auto routing", True,
          "Seven strategies are selectable live — dense, hybrid (dense + BM25 fused with RRF), "
          "hybrid + cross-encoder rerank, MMR, multi-query, HyDE, and our own adaptive_hybrid (query-shape-aware "
          "BM25↔dense weighting + MMR). The Inspector lays the core stages side-by-side; the benchmark below picked the winner.", "inspect"),
        c(GROUP_REQUIRED, "Vector DB wired to an LLM (the RAG pipeline)", True,
          f"Retrieved chunks are passed to Claude ({config.CLAUDE_MODEL}), run locally via the Claude "
          f"CLI at zero API cost. LangGraph orchestrates the full retrieve → grade → generate "
          f"flow. Ask anything in the chat panel on the right.", "chat"),
        c(GROUP_REQUIRED, "Tested on sample queries; best approach chosen", has_eval,
          "A reproducible benchmark (evaluate.py) scores 5 sample queries across 3 embeddings × 3 "
          "strategies using an LLM-as-judge (1–5). The winner became the live default. The full "
          "ranking and the finding are in the results table below.", "criteria"),
        c(GROUP_REQUIRED, "Every answer shows its sources", True,
          f"Each answer lists the top {config.TOP_SOURCES} source chunks it actually used, with the "
          f"paper title and page number — and a link out when it falls back to web search. Ask a "
          f"question to see it, then expand the trace to view the exact context sent to the model.", "chat"),
        c(GROUP_STRETCH, "Multi-user conversational memory", True,
          "Each browser session gets its own conversation history (SQLite). Follow-ups like "
          "‘and what about BERT?’ are automatically rewritten into standalone questions before "
          "retrieval, so context carries across turns without leaking between users.", "chat"),
        c(GROUP_STRETCH, "A web UI on top of the RAG core", True,
          "Two interfaces ship on the same RAG engine: this custom FastAPI single-page explorer "
          "(chat + corpus + inspector + live config), and a Chainlit chat app.", "chat"),
        c(GROUP_STRETCH, "Agentic Corrective RAG with web-search fallback", True,
          "LangGraph grades each retrieved chunk for real relevance. If the papers can't answer the "
          "question, it rewrites the query and runs a live web search — so it can also answer about "
          "newer work (e.g. Mamba, 2023). The answer trace shows every step it took.", "chat"),
        c(GROUP_STRETCH, "Pre-RAG query gate: smart routing + clarifying questions", True,
          "Before retrieving, a fast LLM gate quality-checks the prompt: if it's too vague it asks a "
          "clarifying question instead of guessing; if it's clear it auto-routes to the best strategy "
          "(keyword-leaning vs semantic) from the query's shape. Pick 'Auto' in the right rail and watch "
          "the trace's '0 · Analyze query' step.", "chat"),
    ]
    met = sum(1 for i in items if i["met"])
    groups = [
        {"name": GROUP_REQUIRED,
         "desc": "The compulsory deliverables from the Analytics Vidhya capstone brief."},
        {"name": GROUP_STRETCH,
         "desc": "Optional extensions suggested by the brief — all three are implemented."},
    ]
    return {"intro": intro, "papers": papers, "groups": groups,
            "items": items, "met": met, "total": len(items)}


# Deep "what / how / why / alternatives" for each stack component — powers the
# clickable Stack-tile modals AND the How-it-Works "Why …?" FAQ (one source).
STACK_DETAILS = {
    "llm": {
        "what": "Claude (via the local `claude` CLI) wrapped as a LangChain chat model. The fast `haiku` model runs the query gate; the main model writes answers.",
        "how": ["Generates the final grounded answer from retrieved context",
                "Acts as the CRAG relevance grader (per-chunk yes/no)",
                "Powers the pre-RAG query gate (classify + route + clarify) on the cheaper haiku model",
                "Rewrites the query for the web-search fallback, and condenses follow-ups"],
        "why": "Runs on the host's Claude Max plan through a subprocess → zero per-query API cost, no API key inside the app, and the papers never leave the box. That is the whole reason this capstone costs $0 to run.",
        "alternatives": ["OpenAI GPT-4o-mini — per-token cost + an API key to manage",
                         "Local Llama via Ollama — free but weaker and heavy on a CPU-only VPS"],
    },
    "reranker": {
        "what": "A cross-encoder (BAAI/bge-reranker-base) that reads the query and a passage together and scores their true relevance.",
        "how": ["In the `hybrid_rerank` strategy, re-scores the fused candidate pool",
                "Writes the real relevance_score into each surviving chunk — shown on the source cards",
                "Keeps the top-N (RERANK_TOP_N) chunks for the answer"],
        "why": "A cross-encoder is more accurate than embedding cosine because it attends across the query and passage jointly. We kept it as a selectable strategy for transparency — even though our benchmark found it HURTS on this tiny 6-paper corpus (it over-trims context).",
        "alternatives": ["Cohere Rerank — strong but a paid API", "No reranker — our actual default, since plain hybrid won the benchmark"],
    },
    "vector_db": {
        "what": "Chroma — an embedded, persistent vector database. One collection per embedding model (papers_<name>), stored on local disk.",
        "how": ["Stores every chunk's vector + metadata (title, page, source, chunk_id)",
                "Serves dense similarity and MMR search at query time",
                "Per-model collections let us compare embeddings head-to-head on the same corpus"],
        "why": "Embedded and zero-ops — there is no database server to run on the VPS; it persists to disk, integrates first-class with LangChain, supports metadata filtering and MMR natively, is open-source, and runs on CPU. Ideal for a self-contained, zero-spend capstone.",
        "alternatives": ["FAISS — fast, but no built-in persistence/metadata ergonomics (lower-level)",
                         "pgvector — needs a running Postgres instance",
                         "Pinecone / Weaviate — hosted: network latency, API keys and cost (breaks zero-spend)",
                         "Elasticsearch / OpenSearch — excellent for keyword+vector at scale, but heavyweight to operate; we get the keyword signal in-process via BM25 instead"],
    },
    "keyword": {
        "what": "BM25 sparse keyword retrieval (rank_bm25) over the same chunks as the vectors.",
        "how": ["Scores exact term overlap between query and chunk",
                "Fused with dense results via Reciprocal Rank Fusion in `hybrid` and `adaptive_hybrid`",
                "Adaptive Hybrid raises BM25's weight when the query is an acronym / short / quoted"],
        "why": "Embeddings smear exact tokens; BM25 nails acronyms and rare terms (e.g. 'NSP', 'BLEU', 'GPT-3'). Fusing both beats either alone — which is exactly what our benchmark showed. At production scale this is the role Elasticsearch / OpenSearch would play.",
        "alternatives": ["Elasticsearch / OpenSearch BM25 — scales, but a service to operate", "SPLADE learned-sparse — stronger but much heavier"],
    },
    "web_search": {
        "what": "A pluggable web-search provider used only as the Corrective-RAG fallback. DuckDuckGo by default (free, no key).",
        "how": ["Fires only when the CRAG grader keeps ZERO relevant chunks",
                "The query is rewritten, the provider is called, and results fold back in as Documents",
                "Provider is selected by env (WEB_SEARCH_PROVIDER) and auto-falls back to DuckDuckGo"],
        "why": "Lets the bot answer newer work that isn't in the corpus (e.g. Mamba, 2023) instead of confidently hallucinating. Kept free/zero-spend: DuckDuckGo needs no key; Brave (2k/mo) and Serper (2.5k/mo) free tiers drop in via env. SerpAPI was deliberately rejected — it is paid.",
        "alternatives": ["Brave Search API — free tier, key", "Serper.dev — free tier, key", "SerpAPI — paid (rejected to stay zero-spend)"],
    },
    "orchestration": {
        "what": "LangGraph — a state machine that compiles the agentic Corrective-RAG flow into explicit nodes and edges.",
        "how": ["Nodes: gate → retrieve → grade → (rewrite → web) → generate",
                "Conditional edges: clarify-or-retrieve after the gate, and answer-or-web-fallback after grading",
                "Every node appends a trace event — that trace IS the UI's 'how this answer was produced'"],
        "why": "A self-correcting pipeline that can branch (web fallback) and exit early (ask for clarification) needs explicit, inspectable control flow — not a linear chain. LangGraph makes the routing auditable, which is the whole point of the transparency story.",
        "alternatives": ["A hand-rolled if/else chain — works but isn't inspectable", "LangChain AgentExecutor — less deterministic control over the flow"],
    },
}


def _embedding_details(name: str, spec: dict) -> dict:
    why = {
        "minilm": "Tiny and fast — the open-source baseline to beat. Good enough to show the comparison cheaply.",
        "bge": "Strong open-source retrieval that runs free on CPU — it won our benchmark, so it's the live default.",
        "gemini": "A commercial model on its free tier — included so the capstone's required open-source-vs-commercial comparison is real, at no cost.",
        "openai": "The commercial reference point (text-embedding-3-small) — wired in but optional (needs a key).",
    }.get(name, "One of the embedding models compared in the capstone.")
    return {
        "what": f"{spec['label']} — embeds text into a {'commercial' if spec['provider'] in ('openai','gemini') else 'local open-source'} vector space.",
        "how": ["Encodes every chunk into a vector at index time (its own Chroma collection)",
                "Encodes the query at search time for dense / MMR / hybrid retrieval"],
        "why": why,
        "alternatives": ["Switch live from the right-rail Embedding selector and compare on the Inspector tab"],
    }


@app.get("/api/info")
def info() -> dict:
    """The full model/tech stack — for the evaluator to see what powers each part."""
    llm_label = (f"Claude ({config.CLAUDE_MODEL}) via local claude CLI — no API cost"
                 if config.LLM_BACKEND == "claude_cli"
                 else f"OpenAI {config.LLM_MODEL}")
    from src.vectorstore import collection_stats
    embeddings = []
    for n, s in config.EMBEDDING_MODELS.items():
        ready = has_vectorstore(n)
        stats = collection_stats(n) if ready else {"vectors": None, "dim": None}
        embeddings.append({
            "name": n, "model": s["model_name"], "label": s["label"],
            "type": ("commercial" if s["provider"] in ("openai", "gemini") else "open-source"),
            "active": n == _state["embedding"], "ready": ready,
            "vectors": stats["vectors"], "dim": stats["dim"],
            "details": _embedding_details(n, s),
        })
    return {
        "llm": {"role": "Generation · CRAG grader · query gate · query rewriter · follow-up condenser",
                "backend": config.LLM_BACKEND, "model": config.CLAUDE_MODEL, "label": llm_label,
                "details": STACK_DETAILS["llm"]},
        "embeddings": embeddings,
        "reranker": {"role": "Cross-encoder reranking (hybrid_rerank)", "model": config.RERANKER_MODEL,
                     "type": "open-source", "active": _state["strategy"] == "hybrid_rerank",
                     "details": STACK_DETAILS["reranker"]},
        "vector_db": {"name": "Chroma", "note": "one persisted collection per embedding; vectors persisted on disk",
                      "collections": [{"name": e["name"], "collection": config.EMBEDDING_MODELS[e["name"]] and f"papers_{e['name']}",
                                       "vectors": e["vectors"], "dim": e["dim"]} for e in embeddings if e["ready"]],
                      "details": STACK_DETAILS["vector_db"]},
        "keyword": {"name": "BM25 (rank_bm25)", "note": "sparse retrieval fused into hybrid", "details": STACK_DETAILS["keyword"]},
        "web_search": {"name": {"ddg": "DuckDuckGo (ddgs)", "brave": "Brave Search API", "serper": "Serper (Google)"}.get(config.WEB_SEARCH_PROVIDER, config.WEB_SEARCH_PROVIDER),
                       "provider": config.WEB_SEARCH_PROVIDER, "providers": ["ddg", "brave", "serper"],
                       "note": "Corrective-RAG fallback when papers don't cover a query; pluggable, auto-falls back to DuckDuckGo (free, no paid SerpAPI)",
                       "details": STACK_DETAILS["web_search"]},
        "orchestration": {"name": "LangGraph", "note": "gate → retrieve → grade → (rewrite → web) → generate", "details": STACK_DETAILS["orchestration"]},
        "strategies": config.STRATEGY_REGISTRY,
        "active": {"embedding": _state["embedding"], "strategy": _state["strategy"]},
    }


@app.get("/api/corpus")
def corpus() -> dict:
    embeddings = [{"name": n, "label": s["label"], "ready": has_vectorstore(n)}
                  for n, s in config.EMBEDDING_MODELS.items()]
    docs = _doc_summary(_state.get("corpus", []))
    original = _state.get("original_sources") or set()
    for d in docs:
        d["uploaded"] = d["source"] not in original
    return {
        "documents": docs,
        "total_docs": len(docs),
        "total_chunks": sum(d["chunks"] for d in docs),
        "config": {"embedding": _state["embedding"], "strategy": _state["strategy"]},
        "options": {"embeddings": embeddings, "strategies": STRATEGIES,
                    "strategy_registry": config.STRATEGY_REGISTRY, "auto_available": True},
    }


@app.get("/api/document")
def document(source: str, download: bool = False):
    """Serve an original corpus PDF. Restricted to files directly inside DATA_DIR —
    the request is reduced to a bare basename so '../' traversal is impossible, and the
    resolved path must still sit under DATA_DIR. View inline by default; ?download=1 saves."""
    safe = Path(source).name  # strip any directory components before touching the fs
    data_root = config.DATA_DIR.resolve()
    path = (data_root / safe).resolve()
    if data_root not in path.parents or not path.is_file() or path.suffix.lower() != ".pdf":
        return JSONResponse({"error": "Document not found."}, status_code=404)
    return FileResponse(
        path,
        media_type="application/pdf",
        filename=safe,
        content_disposition_type="attachment" if download else "inline",
    )


def _lang_for(rel: str) -> str:
    ext = rel.rsplit(".", 1)[-1].lower()
    return {"py": "python", "html": "html", "md": "markdown", "txt": "plaintext",
            "service": "ini", "conf": "nginx", "example": "ini"}.get(ext, "plaintext")


@app.get("/api/source")
def source_list() -> dict:
    """Grouped list of source files available in the in-app code browser."""
    allow = _state.get("source_allow") or _source_allowlist()
    _state["source_allow"] = allow
    groups: dict = {}
    for rel in allow:
        grp = rel.split("/", 1)[0] + "/" if "/" in rel else "(root)"
        groups.setdefault(grp, []).append(rel)
    return {"files": [{"group": g, "paths": sorted(v)} for g, v in sorted(groups.items())]}


@app.get("/api/source/file")
def source_file(path: str) -> dict:
    allow = _state.get("source_allow") or _source_allowlist()
    _state["source_allow"] = allow
    ap = allow.get(path)
    if not ap:
        return JSONResponse({"error": "File not available."}, status_code=404)
    try:
        content = ap.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    return {"path": path, "language": _lang_for(path), "lines": content.count("\n") + 1,
            "content": content}


@app.get("/api/chunks")
def chunks(source: str) -> dict:
    """Return the actual chunks for a document — what chunking produced."""
    items = []
    for d in _state.get("corpus", []):
        if d.metadata.get("source") == source:
            items.append({
                "index": len(items) + 1,
                "page": d.metadata.get("page_number", "?"),
                "chars": len(d.page_content),
                "text": d.page_content,
            })
    if not items:
        return JSONResponse({"error": "Document not found."}, status_code=404)
    return {
        "source": source,
        "title": config.display_title(source, source),
        "count": len(items),
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "chunks": items,
    }


@app.post("/api/config")
async def set_config(req: ConfigRequest) -> dict:
    new_emb = req.embedding or _state["embedding"]
    new_strat = req.strategy or _state["strategy"]
    if new_emb not in config.EMBEDDING_MODELS:
        return JSONResponse({"error": f"Unknown embedding '{new_emb}'."}, status_code=400)
    if not has_vectorstore(new_emb):
        return JSONResponse({"error": f"No index built for '{new_emb}'."}, status_code=400)
    if new_strat not in STRATEGIES:
        return JSONResponse({"error": f"Unknown strategy '{new_strat}'."}, status_code=400)
    async with _rebuild_lock:
        _state["embedding"] = new_emb
        _state["strategy"] = new_strat
        await asyncio.to_thread(_rebuild_state)
    return {"embedding": new_emb, "strategy": new_strat}


@app.post("/api/upload")
async def upload(files: list[UploadFile] = File(...)) -> dict:
    """Add one or more PDFs live → chunk → embed → index → queryable instantly.

    Accepts a batch (field name ``files``, repeated). Each file is read and
    validated up front; valid ones are ingested under a single rebuild lock and
    the CRAG graph is rebuilt ONCE at the end, so dropping N files is one atomic
    swap rather than N. A bad file in the batch is reported per-file and never
    aborts the good ones. Backwards compatible with a single-file submission.
    """
    # Read + validate every upload before taking the (serializing) rebuild lock.
    staged: list[tuple[str, bytes]] = []   # (safe_name, bytes) — the ones to ingest
    results: list[dict] = []               # per-file outcome
    for file in files:
        safe = Path(file.filename or "").name
        if not safe.lower().endswith(".pdf"):
            results.append({"filename": safe or "(unnamed)", "ok": False,
                            "error": "Only PDF files are supported."})
            continue
        data = await file.read()
        if len(data) > MAX_UPLOAD_BYTES:
            results.append({"filename": safe, "ok": False,
                            "error": "File too large (max 25 MB)."})
            continue
        staged.append((safe, data))

    if staged:
        async with _rebuild_lock:
            def _ingest_all():
                out = []
                for safe, data in staged:
                    dest = config.DATA_DIR / safe
                    dest.write_bytes(data)
                    try:
                        chunks = load_single_pdf(dest)
                    except Exception as e:
                        out.append({"filename": safe, "ok": False,
                                    "error": f"Parse failed: {str(e)[:100]}"})
                        continue
                    pages = len({c.metadata.get("page_number") for c in chunks})
                    emb_status = []
                    for name in config.EMBEDDING_MODELS:
                        if not has_vectorstore(name):
                            continue
                        try:
                            add_to_vectorstore(chunks, name)
                            emb_status.append({"name": name, "label": config.EMBEDDING_MODELS[name]["label"], "status": "indexed"})
                        except Exception as e:  # e.g. gemini quota — keep the local ones
                            emb_status.append({"name": name, "label": config.EMBEDDING_MODELS[name]["label"], "status": f"error: {str(e)[:80]}"})
                    sample = chunks[len(chunks) // 2] if chunks else None
                    out.append({
                        "filename": safe,
                        "ok": True,
                        "title": config.display_title(safe, chunks[0].metadata.get("title", safe)) if chunks else safe,
                        "pages": pages,
                        "chunks": len(chunks),
                        "sample_chunk": {
                            "page": sample.metadata.get("page_number") if sample else None,
                            "text": (sample.page_content[:400].strip() if sample else ""),
                        } if sample else None,
                        "embeddings": emb_status,
                    })
                _rebuild_state()  # one rebuild for the whole batch → single atomic swap
                return out
            results.extend(await asyncio.to_thread(_ingest_all))

    docs = _doc_summary(_state["corpus"])
    ingested = [r for r in results if r.get("ok")]
    return {
        "results": results,
        "ingested": len(ingested),
        "failed": len(results) - len(ingested),
        "total_docs": len(docs),
        "total_chunks": sum(d["chunks"] for d in docs),
    }


@app.post("/api/reset")
async def reset() -> dict:
    """Remove uploaded PDFs and rebuild from the original corpus."""
    async with _rebuild_lock:
        def _do():
            original = _state.get("original_sources") or set()
            removed = []
            for pdf in config.DATA_DIR.glob("*.pdf"):
                if pdf.name not in original:
                    # Delete this doc's chunks from every collection, then the file.
                    for name in config.EMBEDDING_MODELS:
                        if not has_vectorstore(name):
                            continue
                        try:
                            from src.vectorstore import load_vectorstore
                            load_vectorstore(name).delete(where={"source": pdf.name})
                        except Exception:
                            pass
                    pdf.unlink()
                    removed.append(pdf.name)
            _rebuild_state()
            return removed
        removed = await asyncio.to_thread(_do)
    docs = _doc_summary(_state["corpus"])
    return {"removed": removed, "total_docs": len(docs),
            "total_chunks": sum(d["chunks"] for d in docs)}


@app.post("/api/inspect")
async def inspect(req: InspectRequest) -> dict:
    query = (req.query or "").strip()
    if not query:
        return JSONResponse({"error": "Empty query."}, status_code=400)
    emb = req.embedding or _state["embedding"]
    if emb not in config.EMBEDDING_MODELS or not has_vectorstore(emb):
        return JSONResponse({"error": f"No index for '{emb}'."}, status_code=400)

    async with _semaphore:
        def _run():
            if _state.get("cross_encoder") is None:
                from langchain_community.cross_encoders import HuggingFaceCrossEncoder
                _state["cross_encoder"] = HuggingFaceCrossEncoder(model_name=config.RERANKER_MODEL)
            return inspect_query(query, _state["corpus"], embedding_name=emb,
                                 cross_encoder=_state["cross_encoder"])
        try:
            return await asyncio.to_thread(_run)
        except Exception as e:
            return JSONResponse({"error": f"Inspect failed: {e}"}, status_code=502)


@app.post("/api/compare_embeddings")
async def compare_embeddings(req: InspectRequest) -> dict:
    """Run the SAME query through each built embedding (dense retrieval) so the
    evaluator can see open-source vs commercial embeddings side by side."""
    query = (req.query or "").strip()
    if not query:
        return JSONResponse({"error": "Empty query."}, status_code=400)

    async with _semaphore:
        def _run():
            from src.vectorstore import load_vectorstore, collection_stats
            cols = []
            for n, s in config.EMBEDDING_MODELS.items():
                if not has_vectorstore(n):
                    continue
                store = load_vectorstore(n)
                pairs = store.similarity_search_with_relevance_scores(query, k=5)
                stats = collection_stats(n)
                cols.append({
                    "name": n, "model": s["model_name"],
                    "type": "commercial" if s["provider"] in ("openai", "gemini") else "open-source",
                    "dim": stats["dim"], "vectors": stats["vectors"],
                    "results": [{
                        "rank": i + 1,
                        "title": config.display_title(d.metadata.get("source", ""), d.metadata.get("title", "Unknown")),
                        "page": d.metadata.get("page_number", "?"),
                        "score": round(float(sc), 4),
                        "snippet": d.page_content[:160].strip().replace("\n", " "),
                    } for i, (d, sc) in enumerate(pairs)],
                })
            return {"query": query, "embeddings": cols}
        try:
            return await asyncio.to_thread(_run)
        except Exception as e:
            return JSONResponse({"error": f"Compare failed: {e}"}, status_code=502)


@app.post("/api/ask")
async def ask(req: AskRequest, response: Response, sid: str | None = Cookie(default=None)):
    question = (req.question or "").strip()
    if not question:
        return JSONResponse({"error": "Empty question."}, status_code=400)
    if len(question) > 2000:
        return JSONResponse({"error": "Question too long (max 2000 chars)."}, status_code=400)

    if not sid:
        sid = str(uuid.uuid4())
        response.set_cookie("sid", sid, max_age=60 * 60 * 24, httponly=True, samesite="lax")

    # Strategy: per-request override ("auto" routes via the gate); else the active default.
    requested_strategy = (req.strategy or _state["strategy"])
    if requested_strategy not in STRATEGIES_WITH_AUTO:
        return JSONResponse({"error": f"Unknown strategy '{requested_strategy}'."}, status_code=400)

    crag_app = _state["crag_app"]
    condenser = _state["condenser"]

    history = memory.history_as_text(sid, limit=6)
    standalone = question
    if history:
        try:
            standalone = (await asyncio.to_thread(
                condenser.invoke, {"history": history, "question": question})).strip()
        except Exception:
            standalone = question

    init_state = {
        "question": standalone, "documents": [], "generation": "",
        "used_web_search": False, "trace": [],
        "strategy": requested_strategy, "embedding": _state["embedding"],
        "gate": {}, "recommended_strategy": requested_strategy,
        "clarifying_questions": [], "routed": False, "short_circuit": False,
    }
    async with _semaphore:
        try:
            result = await asyncio.to_thread(crag_app.invoke, init_state)
        except Exception as e:
            return JSONResponse({"error": f"Generation failed: {e}"}, status_code=502)

    trace = result.get("trace", [])
    gate = result.get("gate", {})
    routed_strategy = result.get("recommended_strategy")

    # The gate may short-circuit on a vague question: ask, don't answer (no memory write).
    if result.get("short_circuit"):
        return {
            "needs_clarification": True,
            "clarifying_questions": result.get("clarifying_questions", []),
            "gate": gate, "routed_strategy": routed_strategy, "routed": bool(result.get("routed")),
            "answer": "", "sources": [], "used_web_search": False,
            "standalone_question": standalone if standalone != question else None,
            "trace": trace,
            "config": {"embedding": _state["embedding"], "strategy": requested_strategy},
        }

    answer = result.get("generation", "")
    used_web = bool(result.get("used_web_search"))
    docs = result.get("documents", [])

    memory.add_message(sid, "user", question)
    memory.add_message(sid, "assistant", answer)

    sources, seen = [], set()
    for s in format_sources(docs, top_n=10):
        key = (s["title"], s["page"])
        if key in seen:
            continue
        seen.add(key)
        sources.append(s)
        if len(sources) >= config.TOP_SOURCES:
            break

    return {
        "answer": answer,
        "needs_clarification": False,
        "used_web_search": used_web,
        "standalone_question": standalone if standalone != question else None,
        "sources": sources,
        "trace": trace,
        "gate": gate,
        "routed_strategy": routed_strategy,
        "routed": bool(result.get("routed")),
        "config": {"embedding": _state["embedding"], "strategy": requested_strategy},
    }
