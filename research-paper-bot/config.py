"""
Central configuration for the Research Paper Answer Bot.

Everything tunable lives here so the rest of the codebase never hard-codes
model names, paths, or magic numbers. Values are read from environment
variables (loaded from a .env file) with sensible defaults.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"            # source PDFs go here
STORAGE_DIR = ROOT_DIR / "storage"      # Chroma DBs, SQLite, BM25 caches
CHROMA_DIR = STORAGE_DIR / "chroma"     # one sub-folder per embedding model
SQLITE_PATH = STORAGE_DIR / "chat_memory.db"

for _d in (DATA_DIR, STORAGE_DIR, CHROMA_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Load .env (no error if it is missing; env vars may be set another way)
load_dotenv(ROOT_DIR / ".env")

# ---------------------------------------------------------------------------
# API keys
# ---------------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")

# ---------------------------------------------------------------------------
# LLM backend
# ---------------------------------------------------------------------------
# Generation, the CRAG relevance grader, the query rewriter and the eval judge
# all go through one LLM. Two backends are supported:
#   "claude_cli" -> shell out to the local `claude` CLI (Claude Max, no API $)
#   "openai"     -> ChatOpenAI (needs OPENAI_API_KEY)
LLM_BACKEND = os.getenv("LLM_BACKEND", "claude_cli")

# Claude CLI settings (used when LLM_BACKEND="claude_cli").
CLAUDE_BIN = os.getenv("CLAUDE_BIN", "claude")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "sonnet")   # sonnet | haiku | opus
CLAUDE_TIMEOUT = int(os.getenv("CLAUDE_TIMEOUT", "180"))

# Per-role model overrides. The pre-RAG query gate is a fast classifier, so it
# runs on a cheaper/faster model by default; generation stays on CLAUDE_MODEL.
GATE_MODEL = os.getenv("GATE_MODEL", "haiku")
GRADER_MODEL = os.getenv("GRADER_MODEL", CLAUDE_MODEL)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
# Embedding models compared in the capstone. Keys are friendly names used on
# the CLI; values describe how to build the model. "provider" is either
# "huggingface" (runs locally on the VPS CPU) or "openai" (commercial API).
EMBEDDING_MODELS = {
    "minilm": {
        "provider": "huggingface",
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "label": "MiniLM (open-source, fast baseline)",
    },
    "bge": {
        "provider": "huggingface",
        "model_name": "BAAI/bge-base-en-v1.5",
        "label": "BGE-base (open-source, strong retrieval)",
    },
    "gemini": {
        "provider": "gemini",
        "model_name": "models/gemini-embedding-001",
        "label": "Google gemini-embedding-001 (commercial, free tier)",
    },
    "openai": {
        "provider": "openai",
        "model_name": "text-embedding-3-small",
        "label": "OpenAI text-embedding-3-small (commercial)",
    },
}

# Default embedding model used by the app/index builder.
DEFAULT_EMBEDDING = os.getenv("DEFAULT_EMBEDDING", "bge")

# Human-friendly display titles for the seminal papers (keyed by source filename).
# Used only for presentation so sources read as papers, not filenames — e.g. the
# "GPT-3" paper shows as a citation, not "GPT 3 Language Models Few Shot". Uploaded
# PDFs fall back to their cleaned filename via display_title().
PAPER_TITLES = {
    "Attention_Is_All_You_Need.pdf": "Attention Is All You Need (Vaswani et al., 2017)",
    "BERT.pdf": "BERT (Devlin et al., 2018)",
    "GPT-3_Language_Models_Few_Shot.pdf": "GPT-3: Language Models are Few-Shot Learners (Brown et al., 2020)",
    "RAG_Retrieval_Augmented_Generation.pdf": "RAG: Retrieval-Augmented Generation (Lewis et al., 2020)",
    "Chain_of_Thought_Prompting.pdf": "Chain-of-Thought Prompting (Wei et al., 2022)",
    "InstructGPT_Training_with_Human_Feedback.pdf": "InstructGPT: Training LMs to follow instructions with human feedback (Ouyang et al., 2022)",
}


def display_title(source: str, fallback: str = "") -> str:
    """Nice presentation title for a source filename; falls back for uploads."""
    return PAPER_TITLES.get(source, fallback or source or "Unknown")


# The canonical "original" corpus — the seminal papers shipped with the project.
# "Reset to original" restores exactly these; anything else in data/ (live uploads,
# duplicates) is treated as added and removed on reset. This is the single source
# of truth for what counts as original, so it never drifts with whatever happens
# to be sitting in data/ at startup.
ORIGINAL_SOURCES = set(PAPER_TITLES.keys())

# Cross-encoder reranker (open-source, runs on CPU).
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")

# Generation LLM (commercial, OpenAI). gpt-4o-mini for dev, override for demo.
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
TOP_K = int(os.getenv("TOP_K", "5"))          # candidates fetched per retriever
TOP_SOURCES = int(os.getenv("TOP_SOURCES", "3"))  # sources shown to the user
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "3"))  # kept after reranking

# Retrieval strategy: see STRATEGY_REGISTRY below for the full list.
DEFAULT_STRATEGY = os.getenv("DEFAULT_STRATEGY", "hybrid_rerank")

# Retrieval-strategy registry — the single source of truth shared by the
# dispatcher (src/retrievers.py), the API, and the UI dropdowns. "llm_calls" is
# how many EXTRA LLM calls the strategy itself makes at retrieval time (the CRAG
# grader/generator are separate and not counted here). "auto" is NOT listed —
# it is a routing sentinel resolved by the pre-RAG query gate to one of these.
STRATEGY_REGISTRY = {
    "dense":           {"label": "Dense",           "blurb": "Pure semantic cosine similarity over the Chroma vectors.",                  "cost": "low",    "llm_calls": 0},
    "hybrid":          {"label": "Hybrid",          "blurb": "Dense + BM25 keyword search fused with Reciprocal Rank Fusion.",            "cost": "low",    "llm_calls": 0},
    "hybrid_rerank":   {"label": "Hybrid + Rerank", "blurb": "Hybrid candidates re-scored by a cross-encoder reranker.",                  "cost": "medium", "llm_calls": 0},
    "mmr":             {"label": "MMR",             "blurb": "Max-Marginal-Relevance over the dense store — diversifies, cuts near-duplicates.", "cost": "low", "llm_calls": 0},
    "multi_query":     {"label": "Multi-Query",     "blurb": "An LLM expands the question into several paraphrases, then hybrid-fuses the hits.", "cost": "high", "llm_calls": 1},
    "hyde":            {"label": "HyDE",            "blurb": "An LLM drafts a hypothetical answer; we embed THAT and retrieve against it.", "cost": "high",  "llm_calls": 1},
    "adaptive_hybrid": {"label": "Adaptive Hybrid", "blurb": "Our own: tilts BM25↔dense weights by query shape, then MMR for diversity. No LLM.", "cost": "low", "llm_calls": 0},
}

# Strategy names selectable in the app (the registry keys, in order).
STRATEGY_NAMES = list(STRATEGY_REGISTRY.keys())

# CRAG relevance grading: how many chunks to grade in parallel (each grade is one
# LLM call). Higher = faster grading but more concurrent `claude` subprocesses.
GRADE_CONCURRENCY = int(os.getenv("GRADE_CONCURRENCY", "8"))

# ---------------------------------------------------------------------------
# Live-upload indexing robustness
# ---------------------------------------------------------------------------
# Extra attempts when indexing an uploaded doc into an embedding (commercial
# embeddings like Gemini can rate-limit on big/batched uploads).
UPLOAD_EMBED_RETRIES = int(os.getenv("UPLOAD_EMBED_RETRIES", "2"))
# Skip commercial embeddings (gemini/openai) on live upload so a doc is never
# half-indexed by a quota failure. Off by default (index everywhere, retry+warn).
UPLOAD_SKIP_COMMERCIAL = os.getenv("UPLOAD_SKIP_COMMERCIAL", "false").lower() == "true"

# ---------------------------------------------------------------------------
# Web search (Corrective RAG fallback)
# ---------------------------------------------------------------------------
WEB_SEARCH_RESULTS = int(os.getenv("WEB_SEARCH_RESULTS", "3"))

# Provider for the web-search fallback. "ddg" (DuckDuckGo) is free and needs no
# key; "brave" and "serper" have generous free tiers but need a key. Whatever is
# selected, web_search() auto-falls back to DuckDuckGo if the key is missing or
# the call fails — so the app is never broken by web search. SerpAPI is
# deliberately NOT supported (paid, breaks the zero-spend posture).
WEB_SEARCH_PROVIDER = os.getenv("WEB_SEARCH_PROVIDER", "ddg")  # ddg | brave | serper
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY", "")
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")
WEB_SEARCH_TIMEOUT = int(os.getenv("WEB_SEARCH_TIMEOUT", "10"))


def require_openai_key() -> str:
    """Return the OpenAI key or raise a clear error if it is missing."""
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return OPENAI_API_KEY


def require_gemini_key() -> str:
    """Return the Gemini/Google key or raise a clear error if it is missing."""
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY (or GOOGLE_API_KEY) is not set. Add it to .env to use "
            "the 'gemini' commercial embedding."
        )
    return GEMINI_API_KEY
