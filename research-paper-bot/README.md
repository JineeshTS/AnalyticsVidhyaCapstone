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
  → hybrid + cross-encoder reranker.
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
   DuckDuckGo **web search** before answering.

---

## Project structure

```
research-paper-bot/
├── config.py              # all settings (models, paths, chunking, retrieval)
├── build_index.py         # CLI: ingest PDFs → embed → persist Chroma
├── evaluate.py            # benchmark embeddings × strategies, pick a winner
├── app.py                 # Chainlit conversational UI (entry point)
├── requirements.txt
├── .env.example           # copy to .env and add your OpenAI key
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
    ├── retrievers.py      # dense | hybrid | hybrid_rerank
    ├── rag.py             # basic RAG + source attribution
    ├── crag.py            # Corrective RAG (LangGraph + web-search fallback)
    ├── websearch.py       # DuckDuckGo wrapper (free, no key)
    └── memory.py          # multi-user SQLite chat history
```

---

## Setup (on the VPS or any machine)

```bash
# 1. Python env
python3 -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your key
cp .env.example .env
# edit .env → set OPENAI_API_KEY=sk-...

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

## Compare and choose the best approach

```bash
python evaluate.py
# Runs every embedding × strategy over sample queries, scores answers with an
# LLM judge, reports latency, prints a ranked table, and writes
# storage/eval_results.csv. Use this to justify your final choice to the mentor.
```

Once you've picked a winner, set it as the default in `.env`:
```
DEFAULT_EMBEDDING=bge          # or openai / minilm
DEFAULT_STRATEGY=hybrid_rerank # or dense / hybrid
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
| `DEFAULT_EMBEDDING` | `bge` | `minilm` / `bge` / `openai` |
| `DEFAULT_STRATEGY` | `hybrid_rerank` | `dense` / `hybrid` / `hybrid_rerank` |
| `LLM_MODEL` | `gpt-4o-mini` | Use a stronger model for the final demo |
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
- **Embedding × strategy benchmark** — full `evaluate.py` run (45 combos,
  90 LLM calls); results in `storage/eval_results.csv`. Winner: **bge + hybrid**.
- **Corrective RAG** — in-corpus question answers from papers
  (`used_web_search=False`); out-of-corpus "Mamba (2023)" question correctly
  falls back to **live web search** (`used_web_search=True`, web sources returned).
- **Conversational RAG** — multi-user SQLite isolation verified; follow-up
  condensing rewrites "Who invented it?" into a standalone question using history.
- **Chainlit UI** — boots and serves HTTP 200.

**LLM layer:** runs through the local `claude` CLI (`src/claude_llm.py`) — a custom
LangChain `BaseChatModel` that needs **no OpenAI key** and costs nothing per call.
Set `LLM_BACKEND=openai` + `OPENAI_API_KEY` to use OpenAI instead.
