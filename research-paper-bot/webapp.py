"""
FastAPI web app for the Research Paper Answer Bot.

A thin, self-contained web surface over the existing Corrective-RAG pipeline
(src/crag.py) — built as a cleaner, brandable demo UI alongside the Chainlit app.
It is intentionally light: one /api/ask endpoint plus a single static page.

Production notes (this is meant to run public at paperbot.ganakys.com):
  - The CRAG graph + retriever are built ONCE at startup and reused, so requests
    don't rebuild the BM25 corpus or reload models.
  - A concurrency semaphore caps how many `claude` CLI subprocesses run at once,
    so a public URL can't fan out and drain the Claude Max quota.
  - The blocking CRAG call runs in a worker thread so the event loop stays free.
  - Per-browser session id (cookie) drives the multi-user conversational memory;
    follow-ups are condensed to standalone questions before retrieval.

Run locally:  uvicorn webapp:app --host 127.0.0.1 --port 8011
Behind nginx, access control (Basic Auth + rate limit) is handled at the proxy.
"""

from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Cookie, FastAPI, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel

import config
from src import memory
from src.crag import build_crag_app
from src.rag import format_sources, get_llm

# Reuse the same follow-up condenser the Chainlit app uses.
from langchain_core.prompts import ChatPromptTemplate

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

# Cap concurrent CRAG runs (each can spawn several `claude` calls).
MAX_CONCURRENCY = 2
_semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

STATIC_DIR = Path(__file__).resolve().parent / "web"

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build the heavy objects once.
    _state["crag_app"] = build_crag_app()  # uses DEFAULT_EMBEDDING + DEFAULT_STRATEGY
    _state["condenser"] = CONDENSE_PROMPT | get_llm() | StrOutputParser()
    yield
    _state.clear()


app = FastAPI(title="Research Paper Answer Bot", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "embedding": config.DEFAULT_EMBEDDING,
        "strategy": config.DEFAULT_STRATEGY,
        "backend": config.LLM_BACKEND,
    }


@app.post("/api/ask")
async def ask(req: AskRequest, response: Response, sid: str | None = Cookie(default=None)):
    question = (req.question or "").strip()
    if not question:
        return JSONResponse({"error": "Empty question."}, status_code=400)
    if len(question) > 2000:
        return JSONResponse({"error": "Question too long (max 2000 chars)."}, status_code=400)

    # Per-browser session for multi-user memory.
    if not sid:
        sid = str(uuid.uuid4())
        response.set_cookie("sid", sid, max_age=60 * 60 * 24, httponly=True, samesite="lax")

    crag_app = _state["crag_app"]
    condenser = _state["condenser"]

    # 1. Condense the follow-up against history (conversational RAG).
    history = memory.history_as_text(sid, limit=6)
    standalone = question
    if history:
        try:
            standalone = await asyncio.to_thread(
                condenser.invoke, {"history": history, "question": question}
            )
            standalone = standalone.strip()
        except Exception:
            standalone = question  # condensing is best-effort

    # 2. Run the Corrective-RAG graph (blocking) in a worker thread, rate-limited.
    async with _semaphore:
        try:
            result = await asyncio.to_thread(
                crag_app.invoke,
                {"question": standalone, "documents": [], "generation": "", "used_web_search": False},
            )
        except Exception as e:  # surface a clean error, don't 500 the UI
            return JSONResponse({"error": f"Generation failed: {e}"}, status_code=502)

    answer = result.get("generation", "")
    used_web = bool(result.get("used_web_search"))
    docs = result.get("documents", [])

    # 3. Persist the turn (multi-user safe).
    memory.add_message(sid, "user", question)
    memory.add_message(sid, "assistant", answer)

    # 4. Dedup sources to the top-N for display.
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
    }
