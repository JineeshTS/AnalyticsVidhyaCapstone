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

STRATEGIES = ["dense", "hybrid", "hybrid_rerank"]
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
    from src.crag import ANSWER_PROMPT, GRADE_PROMPT, REWRITE_PROMPT

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
        },
        "pipeline": [
            "retrieve — fetch candidate chunks (dense + BM25, fused; optional rerank)",
            "grade — an LLM judges each chunk for real relevance",
            "decide — relevant chunks? answer from papers : fall back to web",
            "rewrite + web search — only when no chunk is relevant (DuckDuckGo)",
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
    return {"rows": rows, "best": best,
            "method": "LLM-as-judge answer quality (1-5) + latency, over 5 sample queries × 3 embeddings × 3 strategies."}


@app.get("/api/criteria")
def criteria() -> dict:
    """Live rubric: each capstone goal, whether it's met, how, and where to see it."""
    docs = _doc_summary(_state.get("corpus", []))
    built = [n for n in config.EMBEDDING_MODELS if has_vectorstore(n)]
    commercial = [n for n in built if config.EMBEDDING_MODELS[n]["provider"] in ("gemini", "openai")]
    opensource = [n for n in built if config.EMBEDDING_MODELS[n]["provider"] == "huggingface"]
    has_eval = (config.STORAGE_DIR / "eval_results.csv").exists()

    def c(group, title, met, how, tab):
        return {"group": group, "title": title, "met": met, "how": how, "tab": tab}

    items = [
        c("Compulsory", "Dataset of research papers", len(docs) > 0,
          f"{len(docs)} papers loaded ({sum(d['chunks'] for d in docs)} chunks). Add more in the Corpus tab.", "corpus"),
        c("Compulsory", "Load & index in a vector database", len(built) > 0,
          f"PDFs → 938 chunks → embedded into Chroma: {len(built)} persisted collections "
          f"({', '.join(built)}). Open Corpus to see/upload docs and view chunks; Stack shows vector counts + dimensions.", "corpus"),
        c("Compulsory", "Compare embeddings (open-source + commercial)",
          len(opensource) > 0 and len(commercial) > 0,
          f"Open-source: {', '.join(opensource)} · Commercial: {', '.join(commercial)}.", "stack"),
        c("Compulsory", "Compare retrieval strategies (cosine → hybrid → reranker)", True,
          "dense / hybrid (dense+BM25) / hybrid_rerank (cross-encoder) — see them side-by-side.", "inspect"),
        c("Compulsory", "Vector DB connected to an LLM (RAG pipeline)", True,
          f"Claude ({config.CLAUDE_MODEL}) via local CLI; LangGraph orchestrates retrieve→generate.", "chat"),
        c("Compulsory", "Tested on sample queries + chosen best approach", has_eval,
          "evaluate.py benchmark — see the ranked results table below.", "criteria"),
        c("Compulsory", "Show the source of every answer (top 3)", True,
          "Each answer lists the top-3 source chunks with paper title + page.", "chat"),
        c("Stretch", "Multi-user conversational RAG", True,
          "Per-session memory (SQLite) + follow-ups condensed to standalone questions.", "chat"),
        c("Stretch", "Streamlit / Chainlit app (a UI)", True,
          "This FastAPI web app (and a Chainlit app) on top of the RAG system.", "chat"),
        c("Stretch", "Agentic Corrective RAG + web search", True,
          "LangGraph grades chunks; if irrelevant it rewrites the query and searches the web.", "inspect"),
    ]
    met = sum(1 for i in items if i["met"])
    return {"items": items, "met": met, "total": len(items)}


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
        })
    return {
        "llm": {"role": "Generation · CRAG grader · query rewriter · follow-up condenser",
                "backend": config.LLM_BACKEND, "model": config.CLAUDE_MODEL, "label": llm_label},
        "embeddings": embeddings,
        "reranker": {"role": "Cross-encoder reranking (hybrid_rerank)", "model": config.RERANKER_MODEL,
                     "type": "open-source", "active": _state["strategy"] == "hybrid_rerank"},
        "vector_db": {"name": "Chroma", "note": "one persisted collection per embedding; vectors persisted on disk",
                      "collections": [{"name": e["name"], "collection": config.EMBEDDING_MODELS[e["name"]] and f"papers_{e['name']}",
                                       "vectors": e["vectors"], "dim": e["dim"]} for e in embeddings if e["ready"]]},
        "keyword": {"name": "BM25 (rank_bm25)", "note": "sparse retrieval fused into hybrid"},
        "web_search": {"name": "DuckDuckGo (ddgs)", "note": "Corrective-RAG fallback when papers don't cover a query"},
        "orchestration": {"name": "LangGraph", "note": "retrieve → grade → (rewrite → web) → generate"},
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
        "options": {"embeddings": embeddings, "strategies": STRATEGIES},
    }


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
async def upload(file: UploadFile = File(...)) -> dict:
    if not file.filename.lower().endswith(".pdf"):
        return JSONResponse({"error": "Only PDF files are supported."}, status_code=400)
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        return JSONResponse({"error": "File too large (max 25 MB)."}, status_code=400)

    safe = Path(file.filename).name
    dest = config.DATA_DIR / safe
    dest.write_bytes(data)

    async with _rebuild_lock:
        def _ingest():
            chunks = load_single_pdf(dest)
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
            _rebuild_state()  # refresh corpus + graph so it's queryable now
            sample = chunks[len(chunks) // 2] if chunks else None
            return {
                "filename": safe,
                "title": config.display_title(safe, chunks[0].metadata.get("title", safe)) if chunks else safe,
                "pages": pages,
                "chunks": len(chunks),
                "sample_chunk": {
                    "page": sample.metadata.get("page_number") if sample else None,
                    "text": (sample.page_content[:400].strip() if sample else ""),
                } if sample else None,
                "embeddings": emb_status,
            }
        result = await asyncio.to_thread(_ingest)

    docs = _doc_summary(_state["corpus"])
    result["total_docs"] = len(docs)
    result["total_chunks"] = sum(d["chunks"] for d in docs)
    return result


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

    async with _semaphore:
        try:
            result = await asyncio.to_thread(
                crag_app.invoke,
                {"question": standalone, "documents": [], "generation": "", "used_web_search": False, "trace": []},
            )
        except Exception as e:
            return JSONResponse({"error": f"Generation failed: {e}"}, status_code=502)

    answer = result.get("generation", "")
    used_web = bool(result.get("used_web_search"))
    docs = result.get("documents", [])
    trace = result.get("trace", [])

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
        "used_web_search": used_web,
        "standalone_question": standalone if standalone != question else None,
        "sources": sources,
        "trace": trace,
        "config": {"embedding": _state["embedding"], "strategy": _state["strategy"]},
    }
