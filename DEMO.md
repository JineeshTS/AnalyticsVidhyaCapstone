# Demo Guide — Research Paper Answer Bot

A 5–8 minute walkthrough to present to your mentor. It shows every compulsory
goal and all three stretch goals, live.

## 0. One‑time setup (before the demo)

```bash
cd research-paper-bot
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt langchain-google-genai
cp .env.example .env          # set GEMINI_API_KEY; keep LLM_BACKEND=claude_cli

python scripts/download_papers.py        # 6 arXiv papers → data/
python build_index.py --embedding minilm
python build_index.py --embedding bge
python build_index.py --embedding gemini
```

Pre‑build all three indexes **before** the call — the first build downloads the
HF embedding + reranker models (a few hundred MB) and you don't want that running
live. After this, everything is instant.

---

## 1. The pitch (30 sec)

> "A RAG chatbot over 6 seminal GenAI papers. It cites its sources, and when the
> papers don't cover a question it self‑corrects and searches the web. The whole
> LLM layer runs on a local Claude CLI, so it costs nothing per query."

---

## 2. Show the data + index (1 min) — *compulsory: load & index*

```bash
python -m src.ingest
```
Point out: **938 chunks** from 6 PDFs, each chunk carrying `title` + `page_number`
metadata (this is what makes source attribution possible).

---

## 3. Basic RAG with sources (1 min) — *compulsory: RAG + show source*

```bash
python -m src.rag "What is the self-attention mechanism in the Transformer?"
```
Point out: the grounded answer **and** the printed **TOP SOURCES** (paper title +
page). That is the "show the source — top 3" requirement, live.

---

## 4. The embedding × strategy comparison (1–2 min) — *compulsory: experiment & choose*

Open `storage/eval_results.csv` (or re‑run `python evaluate.py`) and walk the
ranked table. Talking points:
- **hybrid (dense + BM25) wins** — beat plain dense on every embedding.
- **the reranker actually hurt** on this small corpus — a real, slightly
  counter‑intuitive finding (it trims to top‑3 and sometimes drops needed context).
- **open‑source `bge` matched commercial Gemini** — so we ship `bge + hybrid`.

This is the "explore approaches and pick the best, with evidence" part the brief
emphasises — having a number behind the choice is what mentors look for.

---

## 5. The Chainlit app (2–3 min) — *stretch 2: UI · stretch 1: conversational · stretch 3: corrective web search*

```bash
chainlit run app.py -w        # opens http://localhost:8000
```

Run these **in order**, in the chat window:

1. **In‑corpus:** *"What is chain‑of‑thought prompting?"*
   → answers from the papers; expand the **source** elements shown under the reply.

2. **Follow‑up (conversational RAG):** *"Who introduced it?"*
   → the bot resolves "it" against the previous turn before retrieving. Mention
   that each browser session has its own isolated history (SQLite) — open a second
   tab to show two independent conversations (**multi‑user**).

3. **Out‑of‑corpus (corrective RAG):** *"What is the Mamba state‑space model from 2023?"*
   → the answer is prefixed **🌐 (answered with web search)** and the sources are
   web links. Explain the LangGraph flow: *retrieve → grade → (irrelevant) →
   rewrite query → web search → generate*.

That single sequence demonstrates all three stretch goals end‑to‑end.

---

## 6. Architecture close (30 sec)

Show `REPORT.md` §3 (the diagram) and make the two design points:
- **Local Claude CLI backend** (`src/claude_llm.py`) → standard LangChain model,
  zero per‑call cost, swappable to OpenAI via one env var.
- **One Chroma collection per embedding** → clean, repeatable comparison.

---

## Quick answers to likely mentor questions

- **"Why did the reranker do worse?"** Small high‑signal corpus; hybrid already
  surfaces the right chunks, and trimming to top‑3 drops context. Rerankers shine
  on large/noisy corpora.
- **"How do you stop hallucination?"** The prompt forces answer‑only‑from‑context;
  CRAG grades chunks and won't answer from weak context — it web‑searches instead.
- **"Is it multi‑user safe?"** Yes — history is keyed by per‑session UUID in SQLite;
  sessions can't see each other's turns (verified).
- **"What would you do next?"** Labelled eval set for precision/recall@k; streaming
  responses; auth on the web search; larger corpus to re‑test the reranker.
```
