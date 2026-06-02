# Demo Guide — Research Paper Answer Bot (RAG Explorer)

A 5–8 minute walkthrough that shows **every compulsory goal and all three
stretch goals**, live — driven almost entirely from the web app itself.

**Live app:** https://paperbot.ganakys.com (HTTP Basic-Auth gated — ask the owner
for the demo credentials). Runs as a systemd service behind nginx; it's always up.

> Prefer to run it locally? See [§ Local run](#local-run) at the bottom. The
> tab-by-tab script below is identical either way.

---

## The app at a glance

A dashboard, not a chat box: a left **nav rail**, a large **workspace**, and a
**docked chat** in the bottom-right that stays visible on every tab.

| Tab | Proves |
|---|---|
| ✅ **Criteria** | Live rubric (10/10 capstone goals) + the real `evaluate.py` results table |
| 📥 **Corpus** | Load & index; **upload a PDF live** and watch chunk → embed → index |
| 🔍 **Inspector** | The 3 retrieval strategies + **how the reranker works**, with scores |
| 🧩 **Stack** | Every model in use (LLM, 3 embeddings, reranker, vector DB, web search) |
| 📄 **Code** | The full source, browsable in-app (read-only, secrets hidden) |
| 💬 **Chat** (docked) | RAG answers with sources · follow-ups · web-search fallback |

---

## The pitch (30 sec)

> "A RAG system over 6 seminal GenAI papers. It cites its sources, and when the
> papers don't cover a question it self-corrects and searches the web. The whole
> LLM layer runs on a **local Claude CLI**, so it costs nothing per query — and
> the app is fully transparent: you can see the retrieval internals, every model,
> the evaluation, and all the code, live."

---

## 1. Criteria tab — meet the rubric head-on (1 min)

Open on the **✅ Criteria** tab. It loads a **10/10** scorecard straight from the
running system (`/api/criteria`), each goal marked met with a one-line *how* and a
**"Show me →"** button that jumps to the tab that proves it.

Scroll to the **Evaluation results** table (read live from `eval_results.csv`):
- **`bge` + `hybrid`** is ★ best (4.8/5, lowest latency).
- Talking point: **hybrid beat plain dense on every embedding**; the **reranker
  actually hurt** on this small corpus; **open-source `bge` matched commercial
  Gemini**. That's the "experiment and choose, with evidence" goal — with numbers.

---

## 2. Corpus tab — load, chunk, index, and add a document live (1–2 min)

Shows the indexed documents (6 papers · 938 chunks). Each chunk carries
`title` + `page_number` — which is what powers source attribution.

**The wow moment:** drag any research PDF onto the drop zone. Watch the live steps:
*parsed N pages → split into M chunks → embedded + indexed into all 3 vector DBs →
ready*, with a sample-chunk preview. Then ask about that paper in the docked chat —
it answers **instantly** with citations. (Hit **↺ Reset** afterward to restore the
original 6.)

This visibly demonstrates *load files → chunk → embed → index in a vector DB*.

---

## 3. Retrieval Inspector — the heart of the comparison (1–2 min)

Type a query (e.g. *"What problem do positional encodings solve?"*) and hit
**Inspect**. Four columns appear side-by-side:

- **Dense** — embedding cosine similarity (semantic), with scores.
- **BM25** — keyword scoring (lexical), with scores.
- **Hybrid** — the two fused (Reciprocal Rank Fusion).
- **Reranked** — the cross-encoder rescoring; **green = survives into the final top-3.**

Point at a chunk that moves position between **Hybrid** and **Reranked** — *that*
reordering is how the reranker works. Switch the embedding dropdown to show the
comparison is real across models.

---

## 4. Chat (docked, bottom-right) — the three stretch goals in one sequence (2 min)

Run these in order in the chat dock:

1. **In-corpus:** *"What is chain-of-thought prompting?"*
   → answers from the papers; **source cards** (title + page) appear under the reply.
2. **Follow-up (conversational RAG):** *"Who introduced it?"*
   → it shows `↳ <rewritten standalone question>` — resolving "it" against history
   before retrieving. Each browser session has isolated history (open a second
   browser/incognito tab to show two independent conversations = **multi-user**).
3. **Out-of-corpus (corrective RAG):** *"What is the Mamba state-space model from 2023?"*
   → the reply gets a **🌐 web search** badge and web-link sources. Explain the
   LangGraph flow: *retrieve → grade → (irrelevant) → rewrite → web search → generate*.

You can also flip the **Embedding / Strategy** dropdowns in the right rail to switch
the live config (`bge`/`hybrid` is the chosen default).

---

## 5. Stack + Code — maximum transparency (1 min)

- **🧩 Stack** — one card per component: **Claude (sonnet)** as the LLM (via local
  CLI, no API cost), the three embeddings (active one highlighted, commercial vs
  open-source tagged), the cross-encoder reranker, Chroma, BM25, DuckDuckGo,
  LangGraph. Answers "which models are you using?" at a glance.
- **📄 Code** — the entire project, browsable with syntax highlighting. Open
  `src/crag.py` (the LangGraph corrective-RAG graph) or `src/claude_llm.py` (the
  local-CLI LLM backend). Read-only and allowlisted — `.env`/secrets are never served.

---

## Quick answers to likely mentor questions

- **"Why did the reranker do worse?"** Small high-signal corpus; hybrid already
  surfaces the right chunks, and trimming to top-3 drops context. Rerankers shine
  on large/noisy corpora. (Show it in the Inspector.)
- **"How do you stop hallucination?"** The prompt answers only from context; CRAG
  grades chunks for real answer-relevance and, if none are relevant, web-searches
  instead of guessing.
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
python evaluate.py                                     # rebuild the results table
```

The legacy Chainlit UI (`chainlit run app.py -w`) still works and also satisfies
the "build a UI" stretch goal; the FastAPI RAG Explorer is the richer demo surface.
