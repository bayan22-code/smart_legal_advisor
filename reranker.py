"""
Reranker using cross-encoder.
Reranks top 10 candidates and returns top 3 most relevant articles.
"""

from typing import Any

from sentence_transformers import CrossEncoder

from config import RERANKER_MODEL, TOP_K_RERANK

try:
    import streamlit as st
except Exception:  # pragma: no cover - streamlit may be unavailable in scripts
    st = None


def _build_reranker_model(model_name: str) -> CrossEncoder:
    return CrossEncoder(model_name)


if st:
    _get_cached_reranker_model = st.cache_resource(show_spinner=False)(_build_reranker_model)
else:
    def _get_cached_reranker_model(model_name: str) -> CrossEncoder:
        return _build_reranker_model(model_name)


class LegalReranker:
    """Cross-encoder reranker for legal relevance."""

    def __init__(self, model_name: str = RERANKER_MODEL, top_k: int = TOP_K_RERANK):
        # Cached once per Streamlit session process
        self.model = _get_cached_reranker_model(model_name)
        self.top_k = top_k

    def rerank(
        self,
        query: str,
        chunks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Rerank chunks by relevance to query.
        Returns top_k most relevant articles.
        """
        if not chunks:
            return []
        pairs = [(query, c["text"]) for c in chunks]
        scores = self.model.predict(pairs)
        indexed = [(scores[i], chunks[i]) for i in range(len(chunks))]
        indexed.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in indexed[: self.top_k]]
