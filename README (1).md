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
  `BAAI/bge-base-en-v1.5` (HuggingFace) **and** commercial `text-embedding-3-small`
  (OpenAI).
- Compares **retrieval strategies**: plain dense cosine → hybrid (dense + BM25)
  → hybrid + cross-encoder reranker.
- Connects the vector DB to an LLM (OpenAI GPT) in a full RAG pipeline.
- Tests on sample queries (`evaluate.py`).
- **Shows the source** of every answer (top-3 context chunks with paper title + page).

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

## Verification status (build sandbox vs. your VPS)

Per a strict no-fabrication policy, here is exactly what was and wasn't tested
when this code was produced:

**Verified in the build sandbox (Python 3.12):**
- All files compile (no syntax errors).
- The full stable dependency set installs and imports together coherently
  (langchain 0.3.30 / langchain-core 0.3.86 / langgraph 0.6.11 / chainlit 2.11.1,
  exact versions pinned in `requirements.txt`).
- All project modules + entry scripts import without error.
- The Chainlit app handlers load.
- The LangGraph Corrective-RAG graph **compiles** with all nodes wired
  (retrieve → grade → [generate | transform_query → web_search → generate]).
- Functional tests pass for the **SQLite multi-user memory** (session isolation)
  and the **source/context formatting** logic.

**NOT runnable in the build sandbox (and why) — runs normally on your VPS:**
- A true end-to-end query: the sandbox has **no OpenAI key** and **no network
  access to OpenAI**, so embeddings/generation could not be executed.
- The HuggingFace open-source embeddings + reranker: `torch` exceeded the
  sandbox disk limit, and HuggingFace model downloads are blocked there. These
  install and run normally on the VPS (the code paths that use them import
  lazily and are wired into verified orchestration).
- PDF ingestion against real files: no sample PDFs in the sandbox.

So: the architecture, wiring, and all network-independent logic are verified.
The first live end-to-end run happens on your VPS once `OPENAI_API_KEY` is set
and `build_index.py` has indexed your PDFs. If anything errors there, send the
traceback and it's a quick fix.
