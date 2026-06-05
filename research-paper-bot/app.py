"""
Chainlit app -- the user-facing conversational RAG interface (stretch goals:
UI + multi-user conversational RAG, sitting on top of the Corrective RAG graph).

Run locally:   chainlit run app.py -w
On the VPS:    chainlit run app.py --host 0.0.0.0 --port 8000

Each browser session gets its own session_id and isolated history (SQLite).
Follow-up questions are condensed against the recent history so pronouns like
"it" / "that paper" resolve correctly before retrieval.
"""

import uuid

import chainlit as cl
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import config
from src import memory
from src.crag import build_crag_app
from src.rag import get_llm

CONDENSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given the conversation history and a follow-up question, rewrite the "
            "follow-up into a standalone question that makes sense on its own. "
            "If it is already standalone, return it unchanged. Return only the "
            "question.",
        ),
        ("human", "History:\n{history}\n\nFollow-up: {question}"),
    ]
)


@cl.on_chat_start
async def on_chat_start():
    session_id = str(uuid.uuid4())
    cl.user_session.set("session_id", session_id)

    # Build the CRAG app once per session (retriever + graph are reused).
    msg = cl.Message(content="")
    await msg.stream_token(
        f"Loading the Research Paper Answer Bot "
        f"(embedding: {config.DEFAULT_EMBEDDING}, strategy: {config.DEFAULT_STRATEGY})...\n"
    )
    crag_app = build_crag_app()
    cl.user_session.set("crag_app", crag_app)
    cl.user_session.set("condenser", CONDENSE_PROMPT | get_llm() | StrOutputParser())

    await msg.stream_token(
        "\nReady. Ask me anything about the indexed Generative AI / LLM papers. "
        "If the papers don't cover it, I'll fall back to a web search."
    )
    await msg.update()


@cl.on_message
async def on_message(message: cl.Message):
    session_id = cl.user_session.get("session_id")
    crag_app = cl.user_session.get("crag_app")
    condenser = cl.user_session.get("condenser")

    # 1. Condense the follow-up against history (conversational RAG).
    history = memory.history_as_text(session_id, limit=6)
    question = message.content
    if history:
        question = await cl.make_async(condenser.invoke)(
            {"history": history, "question": message.content}
        )

    # 2. Run the Corrective RAG graph (blocking call off the event loop).
    thinking = cl.Message(content="Analysing the question…")
    await thinking.send()
    init_state = {
        "question": question, "documents": [], "generation": "",
        "used_web_search": False, "trace": [],
        "strategy": config.DEFAULT_STRATEGY, "embedding": config.DEFAULT_EMBEDDING,
        "gate": {}, "recommended_strategy": config.DEFAULT_STRATEGY,
        "clarifying_questions": [], "routed": False, "short_circuit": False,
    }
    result = await cl.make_async(crag_app.invoke)(init_state)

    # The pre-RAG gate may ask for clarification instead of answering a vague question.
    if result.get("short_circuit"):
        qs = result.get("clarifying_questions") or []
        body = "I can answer that better with a bit more detail:\n\n" + "\n".join(f"- {q}" for q in qs)
        memory.add_message(session_id, "user", message.content)
        await cl.Message(content=body).send()
        return

    answer = result["generation"]
    used_web = result["used_web_search"]
    docs = result["documents"]

    # 3. Persist the turn (per-session, multi-user safe).
    memory.add_message(session_id, "user", message.content)
    memory.add_message(session_id, "assistant", answer)

    # 4. Build source elements (top 3).
    elements = []
    seen = set()
    for d in docs:
        title = d.metadata.get("title", "Unknown")
        page = d.metadata.get("page_number", "?")
        src = d.metadata.get("source", "")
        key = (title, page)
        if key in seen:
            continue
        seen.add(key)
        label = f"{title} — page {page}" if page != "web" else f"{title} (web: {src})"
        elements.append(
            cl.Text(name=label, content=d.page_content[:600], display="inline")
        )
        if len(elements) >= config.TOP_SOURCES:
            break

    prefix = "🌐 (answered with web search)\n\n" if used_web else ""
    await cl.Message(content=prefix + answer, elements=elements).send()


@cl.on_chat_end
def on_chat_end():
    # History stays in SQLite for audit; nothing to tear down.
    pass
