# Research Paper Answer Bot 📚

A **Retrieval-Augmented Generation (RAG)** chatbot over landmark Generative-AI / LLM research papers — *Attention Is All You Need*, *BERT*, *GPT-3*, *RAG*, *Chain-of-Thought*, and *InstructGPT*.

Ask a question and every answer is **grounded in the papers and cited to the page**. If the papers don't cover it, the bot **falls back to a live web search** instead of guessing.

### What's under the hood
- **Agentic Corrective RAG** (LangGraph): retrieve → grade each chunk for real relevance → answer, or rewrite + web-search when the papers fall short.
- **Local Claude** via the `claude` CLI for generation — **$0 per-query API cost**.
- **Conversational memory** — your follow-ups (“and what about BERT?”) are resolved against this session's history, isolated per user.

> This is the Chainlit interface (a capstone stretch goal). The same engine also powers a full FastAPI explorer at **/** — with a retrieval inspector, live config, and a corpus uploader.

**Try one of the starter questions below, or just type your own.**

*Analytics Vidhya · Generative AI Pinnacle Capstone*
