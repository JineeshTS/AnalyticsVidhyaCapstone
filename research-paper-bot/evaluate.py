"""
Evaluation harness -- satisfies the "experiment and choose the best approach"
requirement for both embeddings and retrieval strategies.

Because the sample data has no labelled relevance judgements, this uses two
practical, defensible signals per (embedding x strategy x query):
  - retrieval latency (seconds)
  - an LLM-as-judge answer-quality score (1-5) for relevance + faithfulness

Results are printed as a table and written to storage/eval_results.csv so you
can justify your final choice to your mentor.

Usage:
    python evaluate.py
    python evaluate.py --embeddings bge openai --strategies dense hybrid hybrid_rerank
"""

import argparse
import csv
import time
from statistics import mean

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import config
from src.rag import answer_question, get_llm
from src.retrievers import get_retriever

SAMPLE_QUERIES = [
    "What is the self-attention mechanism and why is it useful?",
    "How does retrieval-augmented generation reduce hallucination?",
    "What problem do positional encodings solve in transformers?",
    "Explain the difference between an encoder and a decoder.",
    "What are the main limitations discussed for large language models?",
]

JUDGE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict evaluator. Given a question and an answer, rate the "
            "answer's relevance and factual grounding on a scale of 1 to 5 "
            "(5 = excellent, directly answers and is well-supported; 1 = poor). "
            "Reply with ONLY the integer.",
        ),
        ("human", "Question: {question}\n\nAnswer: {answer}"),
    ]
)


def judge(question: str, answer: str) -> int:
    chain = JUDGE_PROMPT | get_llm() | StrOutputParser()
    raw = chain.invoke({"question": question, "answer": answer}).strip()
    digits = "".join(c for c in raw if c.isdigit())
    return int(digits[0]) if digits else 0


def run(embeddings, strategies):
    rows = []
    for emb in embeddings:
        for strat in strategies:
            print(f"\n--- Embedding={emb} | Strategy={strat} ---")
            retriever = get_retriever(strat, emb)
            latencies, scores = [], []
            for q in SAMPLE_QUERIES:
                t0 = time.perf_counter()
                result = answer_question(q, retriever=retriever)
                latencies.append(time.perf_counter() - t0)
                score = judge(q, result["answer"])
                scores.append(score)
                print(f"  [{score}/5, {latencies[-1]:.2f}s] {q[:50]}...")
            rows.append(
                {
                    "embedding": emb,
                    "strategy": strat,
                    "avg_score": round(mean(scores), 2),
                    "avg_latency_s": round(mean(latencies), 2),
                }
            )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", nargs="+", default=["minilm", "bge", "gemini"],
                        choices=list(config.EMBEDDING_MODELS))
    parser.add_argument("--strategies", nargs="+",
                        default=["dense", "hybrid", "hybrid_rerank"])
    args = parser.parse_args()

    rows = run(args.embeddings, args.strategies)

    out = config.STORAGE_DIR / "eval_results.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("\n========== RESULTS ==========")
    print(f"{'embedding':<10}{'strategy':<16}{'avg_score':<12}{'avg_latency_s'}")
    for r in sorted(rows, key=lambda x: (-x["avg_score"], x["avg_latency_s"])):
        print(f"{r['embedding']:<10}{r['strategy']:<16}{r['avg_score']:<12}{r['avg_latency_s']}")
    best = max(rows, key=lambda x: (x["avg_score"], -x["avg_latency_s"]))
    print(f"\nRecommended: embedding='{best['embedding']}', strategy='{best['strategy']}'")
    print(f"CSV written to {out}")


if __name__ == "__main__":
    main()
