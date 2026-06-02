"""
Build the vector index.

Usage:
    python build_index.py                      # default embedding (config)
    python build_index.py --embedding minilm   # specific embedding
    python build_index.py --all                # build all embeddings (for comparison)

This loads every PDF in data/, chunks it, embeds it, and persists a Chroma
collection per embedding model.
"""

import argparse

import config
from src.ingest import build_corpus
from src.vectorstore import build_vectorstore


def build_one(embedding_name: str, corpus) -> None:
    print(f"\n=== Building index for embedding: {embedding_name} "
          f"({config.EMBEDDING_MODELS[embedding_name]['label']}) ===")
    build_vectorstore(corpus, embedding_name)
    print(f"Done. Persisted to {config.CHROMA_DIR / embedding_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the RAG vector index.")
    parser.add_argument(
        "--embedding",
        default=config.DEFAULT_EMBEDDING,
        choices=list(config.EMBEDDING_MODELS),
        help="Which embedding model to index with.",
    )
    parser.add_argument(
        "--all", action="store_true", help="Build an index for every embedding model."
    )
    args = parser.parse_args()

    print("Loading and chunking PDFs from data/ ...")
    corpus = build_corpus()
    print(f"Chunked into {len(corpus)} chunks.")

    if args.all:
        for name in config.EMBEDDING_MODELS:
            build_one(name, corpus)
    else:
        build_one(args.embedding, corpus)


if __name__ == "__main__":
    main()
