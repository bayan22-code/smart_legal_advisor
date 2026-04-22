"""
BM25 keyword search for exact legal term matching.
"""

from rank_bm25 import BM25Okapi
import re


def tokenize_arabic(text: str) -> list[str]:
    """Simple tokenizer for Arabic: split on whitespace and punctuation."""
    # Remove punctuation, split on whitespace
    text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)
    return [t for t in text.split() if t]


class BM25Index:
    """BM25 index for Arabic legal chunks."""

    def __init__(self):
        self.index: BM25Okapi | None = None
        self.chunks: list[dict] = []
        self.tokenized: list[list[str]] = []

    def build(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        self.chunks = chunks
        texts = [c["text"] for c in chunks]
        self.tokenized = [tokenize_arabic(t) for t in texts]
        self.index = BM25Okapi(self.tokenized)

    def search(self, query: str, top_k: int = 10) -> list[tuple[dict, float]]:
        """
        Search and return top_k chunks with scores.

        Returns:
            List of (chunk, score) tuples
        """
        if self.index is None:
            return []
        tokens = tokenize_arabic(query)
        scores = self.index.get_scores(tokens)
        indices = scores.argsort()[::-1][:top_k]
        results = []
        for i in indices:
            if scores[i] > 0:
                results.append((self.chunks[i], float(scores[i])))
        return results
