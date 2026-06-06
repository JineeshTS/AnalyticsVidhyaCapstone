# Research Paper Answer Bot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about
Generative AI / LLM research papers, shows its sources, and falls back to a
live web search when the papers don't cover a question.

Built for the **Analytics Vidhya Generative AI Pinnacle** capstone. It covers
**all compulsory goals** and **all three stretch goals**.

---

## What it does (mapped to the capstone brief)

**Compulsory goals**
- Loads PDFs and indexes them in a vector database (Chroma).
- Compares **different embedding models**: open-source `all-MiniLM-L6-v2` and
  `BAAI/bge-base-en-v1.5` (HuggingFace) **and** commercial Google
  `gemini-embedding-001` (free tier). OpenAI `text-embedding-3-small` is also
  wired in and works if an OpenAI key is supplied.
- Compares **retrieval strategies**: plain dense cosine → hybrid (dense + BM25)
  → hybrid + cross-encoder reranker (and four more — see below).
- Connects the vector DB to an LLM in a full RAG pipeline. **The LLM runs through
  a local `claude` CLI backend** (`src/claude_llm.py`) — no paid API, $0 per call.
  Switch to OpenAI any time with `LLM_BACKEND=openai`.
- Tests on sample queries (`evaluate.py`).
- **Shows the source** of every answer (top-3 context chunks with paper title + page).

> See **[../REPORT.md](../REPORT.md)** for the full write-up + evaluation results
> and **[../DEMO.md](../DEMO.md)** for a step-by-step demo script.

**Stretch goals (all three)**
1. **Streamlit/Chainlit app** — a Chainlit chat UI (`app.py`).
2. **Multi-user conversational RAG** — per-session isolated chat history in SQLite,
   with follow-up questions condensed against the conversation.
3. **Agentic Corrective RAG** — a LangGraph state machine that grades retrieved
   chunks and, when they're not relevant, rewrites the query and falls back to a
   **web search** before answering.

**Beyond the brief (post-review enhancements)**
- **Pre-RAG query gate** — before retrieval, a fast (haiku) LLM classifies the
  question, **routes** it to the best strategy by query shape, and **asks a
  clarifying question instead of guessing** when the question is too vague. Runs
  on the cheap model, and fails open. Pick **`auto`** in the UI to let it route.
- **Seven retrieval strategies** — `dense`, `hybrid`, `hybrid_rerank`, `mmr`,
  `multi_query`, `hyde`, and our own **`adaptive_hybrid`** (query-shape-aware
  BM25↔dense weighting + MMR diversity, no extra LLM call). See
  `config.STRATEGY_REGISTRY` / `src/retrievers.py`.
- **Enriched citations** — every answer links each source to the **PDF opened at
  the cited page**, with the cross-encoder relevance score (when reranking).
- **Pluggable web search** — provider layer (`WEB_SEARCH_PROVIDER`): DuckDuckGo
  (free, default), Brave or Serper (free tiers), auto-falling back to DuckDuckGo.
  *SerpAPI is intentionally unsupported — it is paid, breaking the zero-spend goal.*
- **Two UIs, one engine** — the FastAPI explorer embeds the Chainlit chat app
  under its own tab, plus a rebuilt "How it works" (architecture diagram +
  design-rationale FAQ) and clickable stack tiles that explain each choice.
- **Live multi-PDF upload** — drop several PDFs at once; they're chunked,
  embedded and queryable immediately (one atomic index rebuild for the batch).

---

## Project structure

```
research-paper-bot/
├── config.py              # all settings + STRATEGY_REGISTRY (7 strategies)
├── build_index.py         # CLI: ingest PDFs → embed → persist Chroma
├── evaluate.py            # benchmark embeddings × 7 strategies, pick a winner
├── webapp.py              # FastAPI RAG Explorer (primary web UI)
├── web/index.html         # the explorer UI (single page)
├── app.py                 # Chainlit conversational UI (second interface, /chat)
├── chainlit.md            # Chainlit welcome screen (capstone-branded)
├── requirements.txt
├── .env.example           # copy to .env (defaults work; GEMINI_API_KEY optional)
├── data/                  # put research-paper PDFs here
├── storage/               # generated: Chroma DBs + chat_memory.db (git-ignored)
├── scripts/
│   └── download_papers.py # fetch 6 seminal papers from arXiv (optional)
├── deploy/
│   ├── nginx.conf         # reverse proxy for the VPS
│   └── paperbot.service   # systemd unit
└── src/
    ├── ingest.py          # load + chunk PDFs, attach title/page metadata
    ├── embeddings.py      # embedding factory (HF + OpenAI)
    ├── vectorstore.py     # Chroma build/load (one collection per embedding)
    ├── retrievers.py      # 7 strategies (dense…hyde, adaptive_hybrid) + registry
    ├── rag.py             # basic RAG + source attribution
    ├── crag.py            # Corrective RAG (LangGraph) + pre-RAG query gate
    ├── inspect.py         # per-strategy + pipeline-internals views for the Inspector
    ├── websearch.py       # pluggable web search (ddg default | brave | serper)
    └── memory.py          # multi-user SQLite chat history
```

---

## Setup (on the VPS or any machine)

```bash
# 1. Python env
python3 -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure — defaults are fine; generation uses the local `claude` CLI ($0)
cp .env.example .env
# optional: set GEMINI_API_KEY to build/use the commercial `gemini` embedding

# 4. Get papers (either drop your own PDFs in data/, or:)
python scripts/download_papers.py

# 5. Build the index (default embedding = bge)
python build_index.py
#    or build all three for the comparison:
python build_index.py --all
```

> First run downloads the open-source embedding/reranker models from HuggingFace
> (a few hundred MB). This needs outbound internet from the VPS, which it has.

## Run

```bash
# Quick CLI test of the basic RAG pipeline:
python -m src.rag "What is self-attention?"

# CLI test of the Corrective RAG (with web-search fallback):
python -m src.crag "What is retrieval augmented generation?"

# The chat UI (local):
chainlit run app.py -w

# The chat UI (VPS, reachable on port 8000):
chainlit run app.py --host 0.0.0.0 --port 8000
```

### Web app (FastAPI)

A second, brandable UI (`webapp.py` + `web/index.html`) wraps the same
Corrective-RAG pipeline behind a `/api/ask` endpoint — built once at startup,
concurrency-capped so a public URL can't drain the LLM quota.

```bash
uvicorn webapp:app --host 127.0.0.1 --port 8011      # then open http://127.0.0.1:8011
```

For a persistent public deployment behind nginx, see `deploy/paperbot.service`
(systemd) and `deploy/paperbot.nginx.conf` (TLS + rate limits; an optional
Basic-Auth gate can be re-enabled). A live instance runs at
**https://paperbot.ganakys.com** (open for evaluation, rate-limited).

## Compare and choose the best approach

```bash
python evaluate.py
# Runs every embedding × all 7 strategies over 5 sample queries (3×7×5 = 105 runs),
# scores answers with an LLM judge, reports latency, prints a ranked table, and
# writes storage/eval_results.csv. Crash-safe + incremental. Justify your choice
# to the mentor with numbers. (Result: dense ties hybrid at the top; the advanced
# strategies underperform on this small corpus; our adaptive_hybrid is the best of
# them. See ../REPORT.md §5.)
```

Once you've picked a winner, set it as the default in `.env`:
```
DEFAULT_EMBEDDING=bge       # minilm / bge / gemini
DEFAULT_STRATEGY=hybrid     # dense | hybrid | hybrid_rerank | mmr | multi_query | hyde | adaptive_hybrid
                            # (or pick "auto" in the app to let the query gate route per question)
```

---

## Deploy on the VPS (persistent)

```bash
# Put the project at /opt/research-paper-bot, create .venv, pip install, build index.
# Then run it as a service and reverse-proxy it:
sudo cp deploy/paperbot.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now paperbot

# Nginx: edit deploy/nginx.conf (set YOUR_DOMAIN), then
sudo cp deploy/nginx.conf /etc/nginx/sites-available/paperbot
sudo ln -s /etc/nginx/sites-available/paperbot /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
sudo certbot --nginx -d YOUR_DOMAIN   # HTTPS
```

---

## Configuration knobs (`.env` / `config.py`)

| Setting | Default | Meaning |
|---|---|---|
| `DEFAULT_EMBEDDING` | `bge` | `minilm` / `bge` / `gemini` |
| `DEFAULT_STRATEGY` | `hybrid` | one of the 7 strategies (or `auto` in the app) |
| `CLAUDE_MODEL` | `sonnet` | generation model (local `claude` CLI) |
| `GATE_MODEL` | `haiku` | fast model for the pre-RAG query gate |
| `WEB_SEARCH_PROVIDER` | `ddg` | `ddg` / `brave` / `serper` (auto-falls back to ddg) |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 1000 / 150 | Chunking |
| `TOP_K` | 5 | Candidates fetched per retriever |
| `TOP_SOURCES` | 3 | Sources shown to the user |

---

## Verification status — verified end-to-end on the VPS ✅

This project has been **run and verified end-to-end** (Python 3.12, VPS), not just
compiled. Evidence (see `../REPORT.md` §6 for detail):

- **Ingest + index** — 6 arXiv papers → **938 chunks** → 3 Chroma collections
  (minilm, bge, gemini), all built successfully.
- **Basic RAG with sources** — grounded answers citing paper title + page
  (e.g. BERT MLM question → BERT pp. 2/16/3).
- **Embedding × strategy benchmark** — full `evaluate.py` run over all 7 strategies
  (3×7×5 = **105 runs**); results in `storage/eval_results.csv`. `dense` and `hybrid`
  tie for the top strategy average (4.67/5); live default **bge + hybrid**; our
  `adaptive_hybrid` is the best of the advanced strategies (4.55).
- **Corrective RAG** — in-corpus question answers from papers
  (`used_web_search=False`); out-of-corpus "Mamba (2023)" question correctly
  falls back to **live web search** (`used_web_search=True`, web sources returned).
- **Conversational RAG** — multi-user SQLite isolation verified; follow-up
  condensing rewrites "Who invented it?" into a standalone question using history.
- **Chainlit UI** — boots and serves HTTP 200.

**LLM layer:** runs through the local `claude` CLI (`src/claude_llm.py`) — a custom
LangChain `BaseChatModel` that needs **no OpenAI key** and costs nothing per call.
Set `LLM_BACKEND=openai` + `OPENAI_API_KEY` to use OpenAI instead.
