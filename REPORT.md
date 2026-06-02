# Research Paper Answer Bot — Capstone Project Report

**Analytics Vidhya — Generative AI Pinnacle**
**Project type:** 2‑week capstone · **Project title:** Research Paper Answer Bot

---

## 1. Executive summary

This project builds a **Retrieval‑Augmented Generation (RAG)** chatbot that answers
questions about seminal Generative‑AI / LLM research papers, **shows the source**
of every answer, and **falls back to a live web search** when the indexed papers
don't cover a question.

It implements **all compulsory goals** and **all three stretch goals** in the
brief, and — crucially — it has been **run and verified end‑to‑end**, not merely
written. Every result table and behaviour in this report comes from an actual run
on the deployment VPS.

A notable engineering decision: the entire LLM layer runs through a **local
`claude` CLI backend** instead of a paid API, so the system operates at **zero
per‑call API cost** while still presenting a standard LangChain interface. The
"commercial embedding" comparison the brief asks for is satisfied with **Google
Gemini embeddings** (free tier), alongside two open‑source HuggingFace models.

---

## 2. Dataset

Six seminal Generative‑AI / LLM papers were downloaded from arXiv
(`scripts/download_papers.py`) into `data/`:

| Paper | arXiv | Why it matters |
|---|---|---|
| Attention Is All You Need | 1706.03762 | The Transformer / self‑attention |
| BERT | 1810.04805 | Bidirectional pre‑training (MLM + NSP) |
| GPT‑3: Language Models are Few‑Shot Learners | 2005.14165 | In‑context / few‑shot learning |
| Retrieval‑Augmented Generation | 2005.11401 | The RAG pattern itself |
| Chain‑of‑Thought Prompting | 2201.11903 | Step‑by‑step reasoning prompts |
| InstructGPT (RLHF) | 2203.02155 | Instruction tuning with human feedback |

Ingestion (`src/ingest.py`) loads each PDF page‑by‑page (PyMuPDF), attaches
`title` / `source` / `page_number` metadata, and splits the text with a
`RecursiveCharacterTextSplitter` (1000‑char chunks, 150 overlap). **Result: 938
chunks.** The page/title metadata is what powers the "show the source" requirement.

---

## 3. Architecture

```
                 ┌──────────── Chainlit UI (app.py) ────────────┐
 user ──▶ browser│  per-session id · SQLite history · condenser │
                 └───────────────────────┬──────────────────────┘
                                          │ standalone question
                                          ▼
                 ┌──────── Corrective RAG graph (src/crag.py, LangGraph) ───────┐
                 │  retrieve ─▶ grade docs ─┬─ relevant ───────────▶ generate    │
                 │                          └─ not relevant ─▶ rewrite ─▶ web ──▶ │
                 └───────────┬───────────────────────────────────────┬──────────┘
                             │ retriever                              │ LLM
        ┌────────────────────▼─────────────┐         ┌────────────────▼───────────────┐
        │ Retrieval (src/retrievers.py)     │         │ LLM backend (src/claude_llm.py) │
        │  dense | hybrid | hybrid_rerank   │         │  local `claude` CLI (headless)  │
        └───────┬───────────────────────────┘         │  no API key · $0 per call       │
                │ Chroma + BM25 + cross-encoder        └─────────────────────────────────┘
        ┌───────▼───────────────────────────┐
        │ Vector store (src/vectorstore.py)  │  one Chroma collection per embedding
        │  embeddings (src/embeddings.py):   │
        │   minilm · bge (HF) · gemini (API) │
        └────────────────────────────────────┘
```

### 3.1 LLM backend — local Claude CLI (`src/claude_llm.py`)

`ClaudeCLIChat` is a custom `BaseChatModel` that shells out to
`claude -p --output-format json` in headless mode. It:

- sends the prompt on **stdin** (so large RAG contexts pass safely),
- maps system messages to `--system-prompt` plus a guard so the CLI behaves as a
  plain text endpoint (no tool use),
- implements `with_structured_output()` via **JSON‑mode prompting + Pydantic
  validation** (the CLI has no function‑calling surface) — this is what the CRAG
  relevance grader uses.

The backend is config‑switchable (`LLM_BACKEND=claude_cli|openai`), so the project
also runs against OpenAI unchanged if a key is supplied. Default is `claude_cli`.

### 3.2 Embeddings (`src/embeddings.py`)

| Name | Provider | Model | Type |
|---|---|---|---|
| `minilm` | HuggingFace | all‑MiniLM‑L6‑v2 | open‑source, fast baseline |
| `bge` | HuggingFace | BAAI/bge‑base‑en‑v1.5 | open‑source, strong retrieval |
| `gemini` | Google | gemini‑embedding‑001 | **commercial (free tier)** |

Each embedding gets its **own persisted Chroma collection**, so the comparison
runs without re‑indexing or models clobbering each other.

### 3.3 Retrieval strategies (`src/retrievers.py`)

1. **dense** — cosine similarity over the Chroma vectors.
2. **hybrid** — dense **+ BM25** keyword search fused with an `EnsembleRetriever`
   (equal weights / reciprocal‑rank fusion).
3. **hybrid_rerank** — hybrid candidates re‑scored by an open‑source
   **cross‑encoder reranker** (`BAAI/bge‑reranker‑base`) and trimmed to the top‑3.

### 3.4 Source attribution (`src/rag.py`)

Every answer returns the **top‑3 context chunks** with paper title + page number,
and the prompt instructs the model to answer *only* from the provided context and
cite paper titles inline.

---

## 4. Goal coverage

### Compulsory goals — all met
- ✅ **Dataset / load + index** — 6 arXiv papers → 938 chunks → Chroma.
- ✅ **Compare embeddings (open‑source + commercial)** — minilm, bge (HF) **and**
  Gemini (commercial). See §5.
- ✅ **Compare retrieval strategies (cosine → hybrid → reranker)** — dense, hybrid,
  hybrid_rerank. See §5.
- ✅ **Vector DB ↔ LLM RAG pipeline** — `src/rag.py` over the local Claude backend.
- ✅ **Test on sample queries** — see §6.
- ✅ **Show the source (top‑3)** — title + page on every answer.

### Stretch goals — all three met
- ✅ **Multi‑user conversational RAG** — per‑session SQLite history
  (`src/memory.py`) with follow‑up questions condensed to standalone form before
  retrieval.
- ✅ **Chainlit app** — `app.py`, a streaming chat UI.
- ✅ **Agentic Corrective RAG + web search** — `src/crag.py`, a LangGraph state
  machine that grades retrieved chunks and, when they're irrelevant, rewrites the
  query and falls back to a DuckDuckGo web search.

---

## 5. Evaluation — choosing the best approach

### 5.1 Method (`evaluate.py`)

The sample corpus has no labelled relevance judgements, so each
(embedding × strategy × query) combination is scored on two practical, defensible
signals:

- **answer quality** — an **LLM‑as‑judge** rates relevance + factual grounding 1–5;
- **latency** — wall‑clock seconds for retrieval + generation.

Run over **3 embeddings × 3 strategies × 5 sample queries = 45 combinations**
(90 LLM calls). Full per‑query log and `storage/eval_results.csv` are produced
by the script.

### 5.2 Results (ranked)

| Rank | Embedding | Strategy | Avg score (/5) | Avg latency (s) |
|---|---|---|---|---|
| 🥇 | **bge** | **hybrid** | **4.8** | **8.76** |
| 🥈 | minilm | hybrid | 4.8 | 9.51 |
| 🥉 | gemini | hybrid | 4.8 | 9.71 |
| 4 | bge | dense | 4.6 | 9.86 |
| 5 | gemini | dense | 4.6 | 9.89 |
| 6 | minilm | dense | 4.6 | 10.24 |
| 7 | gemini | hybrid_rerank | 4.0 | 11.66 |
| 8 | bge | hybrid_rerank | 4.0 | 11.71 |
| 9 | minilm | hybrid_rerank | 3.6 | 13.29 |

### 5.3 Findings

1. **Strategy matters more than the embedding model here.** Every `hybrid` run
   scored 4.8 and every `dense` run scored 4.6, regardless of embedding. On a
   clean, high‑signal 6‑paper corpus the three embeddings are effectively tied.
2. **Hybrid (dense + BM25) is the sweet spot.** Keyword matching complements
   semantic search for the exact technical terms papers are full of (e.g. "BM25",
   "positional encoding", "MLM"), lifting dense 4.6 → 4.8.
3. **The reranker *hurt* on this corpus.** `hybrid_rerank` occupies the bottom
   three places and is slowest. Compressing to the top‑3 occasionally dropped the
   chunk the answer needed (the positional‑encoding query fell to 2/5).
   **Takeaway:** cross‑encoder reranking pays off on large/noisy corpora, but on a
   small high‑signal one it reduces recall for no quality gain.
4. **Open‑source matched commercial.** `bge` equalled the commercial Gemini
   embedding point‑for‑point while running locally and free — so the open‑source
   choice is justified on both quality and cost.

### 5.4 Final configuration

```
DEFAULT_EMBEDDING = bge       # open-source, top score, lowest latency
DEFAULT_STRATEGY  = hybrid    # dense + BM25; beats both dense and reranker here
LLM_BACKEND       = claude_cli
```

---

## 6. Verification evidence (real runs)

**Basic RAG (bge / hybrid_rerank)** — *"How does BERT's masked language model
pre‑training work?"* → accurate MLM + NSP explanation, sources: **BERT** pages 2,
16, 3.

**CRAG, in‑corpus** — *"What is chain‑of‑thought prompting?"* →
`used_web_search = False`, sources: **Chain of Thought Prompting** pages 24, 18, 6.

**CRAG, out‑of‑corpus** — *"What is the Mamba selective state‑space model (2023)?"*
(not in the 6 papers) → `used_web_search = True`, sources: live **Wikipedia +
arXiv + GitHub** results. The corrective web‑search fallback fires exactly as
designed.

**Conversational RAG** — two sessions kept isolated histories; the follow‑up
*"Who invented it?"* was condensed to *"Who invented the Transformer (the neural
network architecture that uses self‑attention)?"* before retrieval.

**Chainlit UI** — boots and serves HTTP 200.

---

## 7. How to run

```bash
cd research-paper-bot
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt langchain-google-genai

cp .env.example .env          # then set GEMINI_API_KEY (and keep LLM_BACKEND=claude_cli)
python scripts/download_papers.py
python build_index.py --embedding minilm
python build_index.py --embedding bge
python build_index.py --embedding gemini    # commercial embedding (needs GEMINI_API_KEY)

# Try it
python -m src.rag  "What is self-attention?"
python -m src.crag "What is retrieval augmented generation?"
python evaluate.py                            # reproduce the comparison table

# The chat UI
chainlit run app.py -w                        # local
chainlit run app.py --host 0.0.0.0 --port 8000  # VPS
```

> The LLM runs through the local `claude` CLI (Claude Max), so **no OpenAI key is
> required**. To use OpenAI instead, set `LLM_BACKEND=openai` and `OPENAI_API_KEY`.

See **[DEMO.md](DEMO.md)** for a step‑by‑step walkthrough to present to a mentor.

---

## 8. Repository layout

```
research-paper-bot/
├── config.py            # all settings (backend, models, chunking, retrieval)
├── build_index.py       # ingest PDFs → embed → persist Chroma
├── evaluate.py          # benchmark embeddings × strategies → eval_results.csv
├── app.py               # Chainlit conversational UI
├── scripts/download_papers.py
├── deploy/              # nginx.conf + systemd unit for the VPS
└── src/
    ├── claude_llm.py    # ← local Claude CLI LLM backend (this project's adapter)
    ├── ingest.py  embeddings.py  vectorstore.py  retrievers.py
    ├── rag.py     crag.py        websearch.py    memory.py
```
