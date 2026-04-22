"""
Embeddings for Arabic legal passages and queries.
Uses SentenceTransformers with multilingual-e5-large.
"""

from sentence_transformers import SentenceTransformer
import numpy as np

from config import EMBEDDING_MODEL
from text_normalization import clean_text

try:
    import streamlit as st
except Exception:  # pragma: no cover - streamlit may be unavailable in scripts
    st = None


def _build_embedding_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


if st:
    _get_cached_embedding_model = st.cache_resource(show_spinner=False)(_build_embedding_model)
else:
    def _get_cached_embedding_model(model_name: str) -> SentenceTransformer:
        return _build_embedding_model(model_name)


class ArabicEmbedder:
    """Embedder for Arabic legal text with E5 prefix convention."""

    PASSAGE_PREFIX = "passage: "
    QUERY_PREFIX = "query: "

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        # Cached once per Streamlit session process
        self.model = _get_cached_embedding_model(model_name)

    def encode_passages(self, texts: list[str], batch_size: int = 32, show_progress: bool = True):
        """Encode passages with 'passage:' prefix."""
        prefixed = [self.PASSAGE_PREFIX + clean_text(t, apply_reversal_fix=False) for t in texts]
        return self.model.encode(prefixed, batch_size=batch_size, show_progress_bar=show_progress)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query with 'query:' prefix."""
        prefixed = self.QUERY_PREFIX + clean_text(query, apply_reversal_fix=False)
        return self.model.encode([prefixed])[0]
