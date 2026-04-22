"""
Ingestion script: load cleaned legal chunks from chunks.txt, embed, build BM25, store in ChromaDB.
Run: python ingest.py
"""

import pickle
import re
from pathlib import Path
from typing import Any

import chromadb

from config import (
    CHROMA_COLLECTION_NAME,
    DATA_DIR,
    VECTOR_DB_PATH,
)
from embeddings import ArabicEmbedder
from bm25_search import BM25Index
from text_normalization import clean_text

BM25_INDEX_PATH = Path(__file__).resolve().parent / "bm25_index.pkl"
CHUNKS_TXT_PATH = DATA_DIR / "chunks.txt"
CHUNK_DELIMITER = "=================================================="


def _clean_index_text(text: str) -> str:
    # Keep a logical (non-mirrored) normalized Arabic form for embeddings and BM25.
    return clean_text(text, apply_reversal_fix=False)


def _to_western_digits(value: str) -> str:
    return value.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))


def _extract_article_number(article_title: str, fallback: int) -> int:
    normalized = _to_western_digits(article_title)
    match = re.search(r"\d+", normalized)
    if not match:
        return fallback
    try:
        return int(match.group())
    except ValueError:
        return fallback


def load_chunks_from_txt(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Chunks file not found: {path}")
    raw = path.read_text(encoding="utf-8", errors="strict")
    blocks = [b.strip() for b in raw.split(CHUNK_DELIMITER) if b.strip()]
    chunks: list[dict[str, Any]] = []
    for i, block in enumerate(blocks, start=1):
        normalized = _clean_index_text(block)
        if not normalized:
            continue
        lines = [ln.strip() for ln in normalized.splitlines() if ln.strip()]
        article_title = lines[0] if lines else f"المادة {i}"
        article_number = _extract_article_number(article_title, i)
        chunks.append(
            {
                "text": normalized,
                "metadata": {
                    "source": "chunks.txt",
                    "law_name": "القانون السوري",
                    "article_title": article_title,
                    "article_number": article_number,
                    "part_index": 0,
                    "total_parts": 1,
                },
            }
        )
    return chunks


def load_bm25_index(bm25: BM25Index | None = None) -> BM25Index:
    """Load persisted BM25 index from disk."""
    idx = bm25 or BM25Index()
    if BM25_INDEX_PATH.exists():
        with open(BM25_INDEX_PATH, "rb") as f:
            loaded = pickle.load(f)
        idx.index = loaded.index
        idx.chunks = loaded.chunks
        idx.tokenized = loaded.tokenized
    return idx


def run_ingest():
    """Full ingestion pipeline from pre-cleaned chunks.txt."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)

    print("Loading chunks from txt...")
    chunks = load_chunks_from_txt(CHUNKS_TXT_PATH)
    print(f"  -> {len(chunks)} articles")

    print("Creating embeddings...")
    embedder = ArabicEmbedder()
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode_passages(texts, show_progress=True)

    print("Building BM25 index...")
    bm25 = BM25Index()
    bm25.build(chunks)
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25, f)
    print(f"  -> Saved to {BM25_INDEX_PATH}")

    print("Storing in ChromaDB...")
    client = chromadb.PersistentClient(path=str(VECTOR_DB_PATH))
    try:
        client.delete_collection(CHROMA_COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [
        f"art_{c['metadata']['article_number']}_p{c['metadata'].get('part_index', 0)}_{i}"
        for i, c in enumerate(chunks)
    ]
    metadatas = []
    for c in chunks:
        m = {**c["metadata"], "article_number": c["metadata"]["article_number"]}
        metadatas.append(m)

    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=metadatas,
    )
    print("  -> Done.")

    print("\nIngestion complete.")


if __name__ == "__main__":
    run_ingest()
