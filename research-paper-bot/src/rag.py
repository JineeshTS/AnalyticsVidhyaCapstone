"""
Basic RAG pipeline (compulsory goal): retrieve -> build context -> generate
answer with GPT -> return the answer together with the top-N source chunks.

This is the straightforward pipeline. The Corrective RAG graph in crag.py wraps
the same pieces with a relevance check and a web-search fallback.
"""

from typing import Dict, List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever

import config
from src.retrievers import get_retriever

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a research assistant that answers questions about AI and "
            "Data Science research papers. Answer ONLY from the provided context. "
            "If the context does not contain the answer, say you don't know based "
            "on the available papers. Be concise and precise, and cite the paper "
            "titles you used inline.",
        ),
        (
            "human",
            "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:",
        ),
    ]
)


def format_context(docs: List[Document]) -> str:
    """Render retrieved chunks into a numbered context block for the prompt."""
    blocks = []
    for i, d in enumerate(docs, 1):
        title = d.metadata.get("title", "Unknown")
        page = d.metadata.get("page_number", "?")
        blocks.append(f"[{i}] (from '{title}', page {page})\n{d.page_content}")
    return "\n\n".join(blocks)


def format_sources(docs: List[Document], top_n: int = config.TOP_SOURCES) -> List[Dict]:
    """Compact source records for display to the user."""
    sources = []
    for d in docs[:top_n]:
        sources.append(
            {
                "title": d.metadata.get("title", "Unknown"),
                "page": d.metadata.get("page_number", "?"),
                "source": d.metadata.get("source", ""),
                "snippet": d.page_content[:300].strip(),
            }
        )
    return sources


def get_llm() -> BaseChatModel:
    """Return the configured LLM backend (Claude CLI by default, OpenAI optional)."""
    if config.LLM_BACKEND == "claude_cli":
        from src.claude_llm import ClaudeCLIChat

        return ClaudeCLIChat(
            model=config.CLAUDE_MODEL,
            timeout=config.CLAUDE_TIMEOUT,
            claude_bin=config.CLAUDE_BIN,
        )

    if config.LLM_BACKEND == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=config.LLM_MODEL,
            temperature=config.LLM_TEMPERATURE,
            api_key=config.require_openai_key(),
        )

    raise ValueError(
        f"Unknown LLM_BACKEND '{config.LLM_BACKEND}'. Use 'claude_cli' or 'openai'."
    )


def answer_question(
    question: str,
    strategy: str = config.DEFAULT_STRATEGY,
    embedding_name: str = config.DEFAULT_EMBEDDING,
    retriever: BaseRetriever = None,
) -> Dict:
    """
    Run the basic RAG pipeline.

    Returns {"answer": str, "sources": [...], "documents": [Document, ...]}.
    A retriever can be passed in to avoid rebuilding it on every call.
    """
    if retriever is None:
        retriever = get_retriever(strategy, embedding_name)

    docs = retriever.invoke(question)
    context = format_context(docs)

    chain = ANSWER_PROMPT | get_llm() | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})

    return {
        "answer": answer,
        "sources": format_sources(docs),
        "documents": docs,
    }


if __name__ == "__main__":
    import sys

    q = sys.argv[1] if len(sys.argv) > 1 else "What is self-attention?"
    result = answer_question(q)
    print("ANSWER:\n", result["answer"])
    print("\nTOP SOURCES:")
    for s in result["sources"]:
        print(f"  - {s['title']} (page {s['page']})")
