"""
Ingestion: load research-paper PDFs, split them into overlapping chunks, and
attach metadata (source file, title, page) so retrieved chunks can be traced
back to their origin -- which is what powers the "show the source" requirement.
"""

from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config


def _clean_title(path: Path) -> str:
    """Turn a filename into a readable paper title."""
    return path.stem.replace("_", " ").replace("-", " ").strip()


def load_pdfs(data_dir: Path = config.DATA_DIR) -> List[Document]:
    """
    Load every PDF in data_dir into page-level Documents.

    PyMuPDFLoader returns one Document per page with metadata including
    'source' (file path) and 'page' (0-indexed). We add a clean 'title'.
    """
    pdf_paths = sorted(Path(data_dir).glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(
            f"No PDFs found in {data_dir}. Add papers or run "
            f"scripts/download_papers.py first."
        )

    docs: List[Document] = []
    for path in pdf_paths:
        title = _clean_title(path)
        for page_doc in PyMuPDFLoader(str(path)).load():
            page_doc.metadata["title"] = title
            page_doc.metadata["source"] = path.name
            # PyMuPDF pages are 0-indexed; present as human page numbers.
            page_doc.metadata["page_number"] = page_doc.metadata.get("page", 0) + 1
            docs.append(page_doc)
    return docs


def chunk_documents(
    docs: List[Document],
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
) -> List[Document]:
    """Split page Documents into overlapping chunks, preserving metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    # Stable chunk id for citation / dedup.
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    return chunks


def build_corpus(data_dir: Path = config.DATA_DIR) -> List[Document]:
    """Convenience: load + chunk in one call."""
    pages = load_pdfs(data_dir)
    chunks = chunk_documents(pages)
    return chunks


if __name__ == "__main__":
    corpus = build_corpus()
    print(f"Loaded and chunked {len(corpus)} chunks.")
    if corpus:
        sample = corpus[0]
        print("\nSample chunk metadata:", sample.metadata)
        print("\nSample chunk text (first 200 chars):")
        print(sample.page_content[:200])
