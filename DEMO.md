# Demo Guide — Research Paper Answer Bot (RAG Explorer)

A 5–8 minute walkthrough that shows **every compulsory goal and all three
stretch goals**, live — driven almost entirely from the web app itself.

**Live app:** https://paperbot.ganakys.com (open for evaluation, rate-limited —
every query spends the owner's local Claude quota). Runs as a systemd service
behind nginx; it's always up.

> Prefer to run it locally? See [§ Local run](#local-run) at the bottom. The
> tab-by-tab script below is identical either way.

---

## The app at a glance

A dashboard, not a chat box: a left **nav rail**, a large **workspace**, and a
**docked chat** in the bottom-right that stays visible on every tab.

| Tab | Proves |
|---|---|
| ✅ **Criteria** | Live rubric (11/11 capstone goals) + the real `evaluate.py` results table |
| 📥 **Corpus** | Load & index; **upload PDFs live** (multi-file) and watch chunk → embed → index |
| 🔍 **Inspector** | **All 7 retrieval strategies** side-by-side, plus the pipeline internals (incl. the reranker), with scores |
| 🧠 **How it works** | Architecture diagram + a design-rationale FAQ (Why Chroma? Why hybrid?…) + the real prompts |
| 🧩 **Stack** | Every model in use — **click any tile** for what/how/why/alternatives |
| 🤖 **Chainlit** | The second UI (Chainlit) embedded live — same RAG engine |
| 📄 **Code** | The full source, browsable in-app (read-only, secrets hidden) |
| 💬 **Chat** (docked) | RAG answers with sources · a **smart query gate** (routes / clarifies) · follow-ups · web-search fallback |

---

## The pitch (30 sec)

> "A RAG system over 6 seminal GenAI papers. It cites its sources, and when the
> papers don't cover a question it self-corrects and searches the web. The whole
> LLM layer runs on a **local Claude CLI**, so it costs nothing per query — and
> the app is fully transparent: you can see the retrieval internals, every model,
> the evaluation, and all the code, live."

---

## 1. Criteria tab — meet the rubric head-on (1 min)

Open on the **✅ Criteria** tab. It loads an **11/11** scorecard straight from the
running system (`/api/criteria`), each goal marked met with a one-line *how* and a
**"Show me →"** button that jumps to the tab that proves it.

Scroll to the **Evaluation results** table (read live from `eval_results.csv` —
**3 embeddings × 7 strategies × 5 queries = 105 runs**):
- **`dense` and `hybrid` tie for the top** strategy average (4.67/5); the live
  default is **`bge` + `hybrid`** (open-source, zero-cost, and more robust than
  pure dense because BM25 also catches exact keywords/acronyms).
- Talking points (all in the auto-generated "finding"): the **advanced strategies
  underperform** here — cross-encoder reranker, Multi-Query and HyDE all score
  *below* the simple baselines on a small high-signal corpus; **our own
  `adaptive_hybrid` is the best of the advanced strategies** (4.55, 3rd overall);
  and **open-source `bge` matched commercial Gemini**. That's "experiment and
  choose, with evidence" — with numbers, and an honest result.

---

## 2. Corpus tab — load, chunk, index, and add a document live (1–2 min)

Shows the indexed documents (6 papers · 938 chunks). Each chunk carries
`title` + `page_number` — which is what powers source attribution.

**The wow moment:** drag one or more research PDFs onto the drop zone (multi-file
supported). Watch the live steps: *parsed N pages → split into M chunks → embedded +
indexed into all 3 vector DBs → ready*, with a sample-chunk preview. Then ask about
that paper in the docked chat — it answers **instantly** with citations. (Hit
**↺ Reset to original 6** afterward — the button only appears when you've added
docs, and restores exactly the canonical 6 papers.)

This visibly demonstrates *load files → chunk → embed → index in a vector DB*.

---

## 3. Retrieval Inspector — the heart of the comparison (1–2 min)

Type a query (e.g. *"What problem do positional encodings solve?"*). Three views:

- **🧭 Compare all 7 strategies** (default) — one column per strategy: dense,
  hybrid, hybrid_rerank, MMR, Multi-Query, HyDE, and our **adaptive_hybrid**, each
  with a cost / LLM-call badge. This is the *strategy* comparison the mentors asked
  for. (Multi-Query and HyDE call the LLM, so it takes ~10–20s.)
- **🔬 Pipeline internals** — how the hybrid+rerank path is built: **Dense → BM25 →
  Hybrid → Reranked** side-by-side with scores; **green = survives into the final
  top-3.** Point at a chunk that moves between **Hybrid** and **Reranked** — *that*
  reordering is how the cross-encoder works.
- **⇄ Compare embeddings** — the same query through every embedding, to show the
  comparison is real across models.

---

## 4. Chat (docked, bottom-right) — the three stretch goals in one sequence (2 min)

Run these in order in the chat dock:

1. **In-corpus:** *"What is chain-of-thought prompting?"*
   → answers from the papers; **source cards** (title + page, with a *View PDF* link
   to the exact page) appear under the reply.
2. **Follow-up (conversational RAG):** *"Who introduced it?"*
   → it shows `↳ <rewritten standalone question>` — resolving "it" against history
   before retrieving. Each browser session has isolated history (open a second
   browser/incognito tab to show two independent conversations = **multi-user**).
3. **Out-of-corpus (corrective RAG):** *"What is the Mamba state-space model from 2023?"*
   → the reply gets a **🌐 web search** badge and web-link sources. Explain the
   LangGraph flow: *gate → retrieve → grade → (irrelevant) → rewrite → web search → generate*.

**Show the smart query gate (our differentiator).** Set the right-rail **Strategy**
to **⚡ Auto (smart routing)**, then:
- Ask a vague question — *"tell me about models"* → instead of guessing, it **asks a
  clarifying question** (no retrieval, no hallucination). Expand the trace to see the
  **"0 · Analyze query"** step.
- Ask a sharp one — *"GPT-3 few-shot?"* → it **auto-routes** to a keyword-leaning
  strategy and answers, showing a **🧭 routed → …** badge. A long conceptual question
  routes differently. That's "route by prompt quality" — live.

You can also flip the **Embedding / Strategy** dropdowns in the right rail to switch
the live config (`bge` / `hybrid` is the chosen default; all 7 strategies + Auto are
selectable).

---

## 5. How it works, Stack, Chainlit + Code — maximum transparency (1–2 min)

- **🧠 How it works** — an **architecture diagram** of the pipeline plus a
  **design-rationale FAQ** that pre-answers the hard questions: *Why Chroma DB?*
  *Why hybrid?* *Why did the reranker hurt?* *How is grounding guaranteed?* — with
  (?) tooltips on every term and the **real prompts** (collapsible). Open this when
  a mentor asks "why did you choose X".
- **🧩 Stack** — one card per component: **Claude (sonnet)** as the LLM (via local
  CLI, no API cost), the three embeddings (active highlighted, commercial vs
  open-source tagged), the cross-encoder reranker, Chroma, BM25, web search,
  LangGraph. **Click any tile** → a modal with *what it is / how we use it / why we
  chose it / alternatives we rejected* (the Chroma tile is the full "Why Chroma?" answer).
- **🤖 Chainlit** — the second UI (the original Chainlit stretch-goal app) embedded
  live in its own tab, on the **same** Corrective-RAG engine. Also open directly at
  **/chat**. Shows "one engine, two interfaces".
- **📄 Code** — the entire project, browsable with syntax highlighting. Open
  `src/crag.py` (the LangGraph corrective-RAG graph + query gate) or
  `src/claude_llm.py` (the local-CLI LLM backend). Read-only and allowlisted —
  `.env`/secrets are never served.

---

## Quick answers to likely mentor questions

- **"Why did the reranker (and Multi-Query / HyDE) do worse?"** Small high-signal
  corpus; hybrid already surfaces the right chunks, so reranking trims context and
  query-expansion dilutes it — both add latency for no gain. They shine on
  large/noisy corpora. (Show it in the Inspector's "Compare all 7 strategies".)
- **"What's your own strategy?"** `adaptive_hybrid` — it reads the query's shape
  (acronym/short → lean BM25; long/conceptual → lean dense) and weights the fusion
  accordingly, then MMR for diversity, with **no extra LLM call**. It was the best
  of the advanced strategies in the benchmark (4.55/5).
- **"Can it route by prompt quality / ask before answering?"** Yes — the pre-RAG
  **query gate** (`Auto` mode) classifies the question, routes the strategy, and
  asks a clarifying question when it's too vague. (Demo'd in step 4.)
- **"Why Chroma?"** Embedded, zero-ops, persists locally, one collection per
  embedding, native metadata + MMR, free. Full comparison in the Stack → Vector DB
  tile (vs FAISS / pgvector / Pinecone / Elasticsearch).
- **"How do you stop hallucination?"** The prompt answers only from context; CRAG
  grades chunks for real answer-relevance and, if none are relevant, web-searches
  instead of guessing; every answer cites its exact source + page.
- **"Which models?"** Stack tab: Claude sonnet (LLM), MiniLM + BGE (open-source)
  + Gemini (commercial) embeddings, BGE cross-encoder reranker.
- **"Is it multi-user safe?"** Yes — per-session UUID history in SQLite; sessions
  can't see each other's turns.
- **"No OpenAI key?"** Correct — `LLM_BACKEND=claude_cli` routes generation through
  the local Claude CLI at zero per-call cost. Set `LLM_BACKEND=openai` to switch.
- **"What next?"** Labelled eval set for precision/recall@k; streaming responses;
  a larger corpus to re-test the reranker.

---

## <a name="local-run"></a>Local run (optional)

```bash
cd research-paper-bot
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt langchain-google-genai
cp .env.example .env          # set GEMINI_API_KEY; keep LLM_BACKEND=claude_cli

python scripts/download_papers.py            # 6 arXiv papers → data/
python build_index.py --embedding minilm     # build all three indexes
python build_index.py --embedding bge
python build_index.py --embedding gemini

uvicorn webapp:app --host 127.0.0.1 --port 8011   # open http://127.0.0.1:8011
```

Pre-build the indexes **before** demoing — the first build downloads the HF
embedding + reranker models (a few hundred MB). After that, everything is instant.

CLI sanity checks (no UI):
```bash
python -m src.ingest                                   # 938 chunks, with metadata
python -m src.rag  "What is self-attention?"           # RAG + top-3 sources
python -m src.crag "What is retrieval augmented generation?"   # corrective RAG
python evaluate.py                                     # rebuild the results table (105 runs, all 7 strategies)
```

The **Chainlit UI** (`chainlit run app.py`) is the second interface and a stretch
goal in its own right — same Corrective-RAG engine, branded for the capstone; it's
embedded live in the explorer's **🤖 Chainlit** tab and served at **/chat**. The
FastAPI RAG Explorer is the richer demo surface.
