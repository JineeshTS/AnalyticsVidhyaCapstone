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

# Retrieval strategy: "dense" | "hybrid" | "hybrid_rerank"
DEFAULT_STRATEGY = os.getenv("DEFAULT_STRATEGY", "hybrid_rerank")

# ---------------------------------------------------------------------------
# Web search (Corrective RAG fallback)
# ---------------------------------------------------------------------------
WEB_SEARCH_RESULTS = int(os.getenv("WEB_SEARCH_RESULTS", "3"))


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
