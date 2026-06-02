"""
Download a starter set of seminal Generative AI / LLM papers from arXiv into
data/. You can also just drop your own PDFs into data/ and skip this.

Usage:
    python scripts/download_papers.py
"""

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# (arXiv id, friendly filename)
PAPERS = [
    ("1706.03762", "Attention_Is_All_You_Need"),
    ("1810.04805", "BERT"),
    ("2005.14165", "GPT-3_Language_Models_Few_Shot"),
    ("2005.11401", "RAG_Retrieval_Augmented_Generation"),
    ("2201.11903", "Chain_of_Thought_Prompting"),
    ("2203.02155", "InstructGPT_Training_with_Human_Feedback"),
]


def download(arxiv_id: str, name: str) -> None:
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    dest = DATA_DIR / f"{name}.pdf"
    if dest.exists():
        print(f"  exists, skipping: {dest.name}")
        return
    print(f"  downloading {url} -> {dest.name}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as f:
        f.write(resp.read())


def main() -> None:
    print(f"Downloading {len(PAPERS)} papers into {DATA_DIR} ...")
    for arxiv_id, name in PAPERS:
        try:
            download(arxiv_id, name)
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED {arxiv_id}: {e}", file=sys.stderr)
    print("Done.")


if __name__ == "__main__":
    main()
