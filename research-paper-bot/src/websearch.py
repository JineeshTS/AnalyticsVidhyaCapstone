"""
Web search wrapper for the Corrective RAG fallback.

Uses DuckDuckGo via the `ddgs` package -- free, no API key required. Results
are returned as LangChain Documents so they slot into the same pipeline as
retrieved paper chunks.
"""

from typing import List

from langchain_core.documents import Document

import config


def web_search(query: str, max_results: int = config.WEB_SEARCH_RESULTS) -> List[Document]:
    """Search the web and return results as Documents (source = the URL)."""
    try:
        from ddgs import DDGS
    except ImportError:  # older package name fallback
        from duckduckgo_search import DDGS  # type: ignore

    docs: List[Document] = []
    with DDGS() as ddgs:
        for hit in ddgs.text(query, max_results=max_results):
            body = hit.get("body", "")
            title = hit.get("title", "")
            url = hit.get("href", "")
            docs.append(
                Document(
                    page_content=f"{title}\n{body}",
                    metadata={
                        "title": title or "Web result",
                        "source": url,
                        "page_number": "web",
                        "origin": "web_search",
                    },
                )
            )
    return docs
