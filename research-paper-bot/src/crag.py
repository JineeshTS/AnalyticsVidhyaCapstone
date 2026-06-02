"""
Agentic Corrective RAG (CRAG) -- stretch goal.

A LangGraph state machine that makes retrieval self-correcting:

    retrieve -> grade each doc for relevance
        |-- all relevant            -> generate
        |-- some/none relevant      -> rewrite query -> web search -> generate

This means when the indexed papers don't cover a question well, the bot falls
back to a live web search instead of confidently answering from weak context.
"""

from typing import List, TypedDict

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

import config
from src.rag import ANSWER_PROMPT, format_context, format_sources, get_llm
from src.retrievers import get_retriever
from src.websearch import web_search


# --- Document relevance grader (structured output) -------------------------
class GradeDocuments(BaseModel):
    """Binary relevance score for a retrieved document."""

    binary_score: str = Field(description="'yes' if relevant to the question, else 'no'")


GRADE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You grade whether a retrieved document is relevant to a user's "
            "question. If it contains keywords or meaning related to the "
            "question, grade it 'yes'. Otherwise 'no'. Be lenient -- the goal "
            "is to filter out clearly irrelevant chunks, not to be strict.",
        ),
        ("human", "Document:\n{document}\n\nQuestion: {question}"),
    ]
)

# --- Query rewriter for the web-search fallback ----------------------------
REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the user's question into a concise, keyword-rich web search "
            "query that would find authoritative information on the topic. "
            "Return only the rewritten query.",
        ),
        ("human", "{question}"),
    ]
)


class CRAGState(TypedDict):
    question: str
    documents: List[Document]
    generation: str
    used_web_search: bool


def _grader():
    return GRADE_PROMPT | get_llm().with_structured_output(GradeDocuments)


# --- Graph nodes -----------------------------------------------------------
def _make_nodes(retriever: BaseRetriever):
    grader = _grader()
    rewriter = REWRITE_PROMPT | get_llm() | StrOutputParser()
    generator = ANSWER_PROMPT | get_llm() | StrOutputParser()

    def retrieve(state: CRAGState) -> CRAGState:
        docs = retriever.invoke(state["question"])
        return {**state, "documents": docs, "used_web_search": False}

    def grade_documents(state: CRAGState) -> CRAGState:
        kept: List[Document] = []
        for d in state["documents"]:
            score = grader.invoke(
                {"document": d.page_content, "question": state["question"]}
            )
            if score.binary_score.strip().lower() == "yes":
                kept.append(d)
        return {**state, "documents": kept}

    def transform_query(state: CRAGState) -> CRAGState:
        better = rewriter.invoke({"question": state["question"]})
        return {**state, "question": better}

    def do_web_search(state: CRAGState) -> CRAGState:
        web_docs = web_search(state["question"])
        # Combine any surviving paper chunks with fresh web results.
        return {
            **state,
            "documents": state["documents"] + web_docs,
            "used_web_search": True,
        }

    def generate(state: CRAGState) -> CRAGState:
        context = format_context(state["documents"])
        answer = generator.invoke(
            {"context": context, "question": state["question"]}
        )
        return {**state, "generation": answer}

    return retrieve, grade_documents, transform_query, do_web_search, generate


def _decide_after_grading(state: CRAGState) -> str:
    """If no relevant paper chunks survived, fall back to web search."""
    return "generate" if state["documents"] else "transform_query"


def build_crag_app(
    strategy: str = config.DEFAULT_STRATEGY,
    embedding_name: str = config.DEFAULT_EMBEDDING,
    retriever: BaseRetriever = None,
):
    """Compile and return the CRAG LangGraph app."""
    if retriever is None:
        retriever = get_retriever(strategy, embedding_name)

    retrieve, grade_documents, transform_query, do_web_search, generate = _make_nodes(
        retriever
    )

    g = StateGraph(CRAGState)
    g.add_node("retrieve", retrieve)
    g.add_node("grade_documents", grade_documents)
    g.add_node("transform_query", transform_query)
    g.add_node("web_search", do_web_search)
    g.add_node("generate", generate)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "grade_documents")
    g.add_conditional_edges(
        "grade_documents",
        _decide_after_grading,
        {"generate": "generate", "transform_query": "transform_query"},
    )
    g.add_edge("transform_query", "web_search")
    g.add_edge("web_search", "generate")
    g.add_edge("generate", END)

    return g.compile()


def answer_question_crag(question: str, app=None, **kwargs) -> dict:
    """Run the CRAG graph and return answer + sources in the same shape as rag.py."""
    if app is None:
        app = build_crag_app(**kwargs)
    final = app.invoke({"question": question, "documents": [], "generation": "",
                         "used_web_search": False})
    return {
        "answer": final["generation"],
        "sources": format_sources(final["documents"]),
        "documents": final["documents"],
        "used_web_search": final["used_web_search"],
    }


if __name__ == "__main__":
    import sys

    q = sys.argv[1] if len(sys.argv) > 1 else "What is retrieval augmented generation?"
    res = answer_question_crag(q)
    print("ANSWER:\n", res["answer"])
    print("\nUsed web search:", res["used_web_search"])
    for s in res["sources"]:
        print(f"  - {s['title']} (page {s['page']})")
