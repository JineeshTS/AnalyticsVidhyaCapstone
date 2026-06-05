"""
Web search wrapper for the Corrective RAG fallback.

Pluggable provider layer. The default is DuckDuckGo (`ddgs` package) — free, no
API key required. Two key-based providers with generous *free* tiers are also
supported and can be enabled purely via env (no code change):

  WEB_SEARCH_PROVIDER = ddg | brave | serper
  BRAVE_API_KEY  = ...        # Brave Search API (2,000 queries/month free)
  SERPER_API_KEY = ...        # Serper.dev Google API (2,500 queries free)

Whatever provider is selected, web_search() automatically falls back to
DuckDuckGo if the key is missing or the call fails, and returns [] only if even
that fails — web search is itself a fallback path, so it must never raise.

SerpAPI is intentionally NOT supported: it is paid, which breaks the project's
zero-spend posture.

Results are returned as LangChain Documents (so they slot into the same pipeline
as retrieved paper chunks); each carries metadata["provider"] for the trace.
"""

from typing import Callable, Dict, List

from langchain_core.documents import Document

import config


def _doc(title: str, body: str, url: str, provider: str) -> Document:
    return Document(
        page_content=f"{title}\n{body}".strip(),
        metadata={
            "title": title or "Web result",
            "source": url,
            "page_number": "web",
            "origin": "web_search",
            "provider": provider,
        },
    )


def _ddg(query: str, n: int) -> List[Document]:
    """DuckDuckGo (free, no key). One retry — ddgs occasionally rate-limits."""
    try:
        from ddgs import DDGS
    except ImportError:  # older package name
        from duckduckgo_search import DDGS  # type: ignore

    last_err = None
    for attempt in range(2):
        try:
            with DDGS() as ddgs:
                return [
                    _doc(h.get("title", ""), h.get("body", ""), h.get("href", ""), "ddg")
                    for h in ddgs.text(query, max_results=n)
                ]
        except Exception as e:  # transient rate-limit / network — retry once
            last_err = e
    if last_err:
        raise last_err
    return []


def _brave(query: str, n: int) -> List[Document]:
    """Brave Search API (free tier, key required)."""
    import requests

    if not config.BRAVE_API_KEY:
        raise RuntimeError("BRAVE_API_KEY not set")
    r = requests.get(
        "https://api.search.brave.com/res/v1/web/search",
        headers={"X-Subscription-Token": config.BRAVE_API_KEY, "Accept": "application/json"},
        params={"q": query, "count": n},
        timeout=config.WEB_SEARCH_TIMEOUT,
    )
    r.raise_for_status()
    results = r.json().get("web", {}).get("results", [])[:n]
    return [_doc(it.get("title", ""), it.get("description", ""), it.get("url", ""), "brave") for it in results]


def _serper(query: str, n: int) -> List[Document]:
    """Serper.dev Google Search API (free tier, key required)."""
    import requests

    if not config.SERPER_API_KEY:
        raise RuntimeError("SERPER_API_KEY not set")
    r = requests.post(
        "https://google.serper.dev/search",
        headers={"X-API-KEY": config.SERPER_API_KEY, "Content-Type": "application/json"},
        json={"q": query, "num": n},
        timeout=config.WEB_SEARCH_TIMEOUT,
    )
    r.raise_for_status()
    results = r.json().get("organic", [])[:n]
    return [_doc(it.get("title", ""), it.get("snippet", ""), it.get("link", ""), "serper") for it in results]


_PROVIDERS: Dict[str, Callable[[str, int], List[Document]]] = {
    "ddg": _ddg,
    "brave": _brave,
    "serper": _serper,
}


def web_search(query: str, max_results: int = config.WEB_SEARCH_RESULTS) -> List[Document]:
    """Search the web and return results as Documents (source = the URL).

    Uses config.WEB_SEARCH_PROVIDER, auto-falling back to DuckDuckGo on any
    failure. Never raises — returns [] in the worst case.
    """
    primary = config.WEB_SEARCH_PROVIDER if config.WEB_SEARCH_PROVIDER in _PROVIDERS else "ddg"
    try:
        docs = _PROVIDERS[primary](query, max_results)
        if docs:
            return docs
    except Exception:
        pass

    # Fall back to the always-free DuckDuckGo provider.
    if primary != "ddg":
        try:
            return _ddg(query, max_results)
        except Exception:
            return []
    return []
