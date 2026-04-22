"""
Configuration for the Arabic Legal RAG system.
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
VECTOR_DB_PATH = BASE_DIR / "vector_db"
PDF_PATH = DATA_DIR / "civil_law.pdf"

# Embedding model
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
EMBEDDING_DIM = 1024  # multilingual-e5-large dimension

# Reranker model
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# ChromaDB
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "syrian_laws_rag")

# OpenRouter - base URL (OpenAI-compatible, client appends /chat/completions)
# Direct endpoint: https://openrouter.ai/api/v1/chat/completions
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct")

# Agentic routing labels
LEGAL_DATABASES = {
    "civil": "Civil Law",
    "penal": "Penal Law",
    "personal_status": "Personal Status Law",
    "all": "All Laws",
}

# Ambiguity handling
AMBIGUITY_ROUTE = "clarify"

# Retrieval
TOP_K_VECTOR = 2
TOP_K_BM25 = 2
TOP_K_HYBRID = 2
TOP_K_RERANK = 2

# Legal article pattern (supports Western + Arabic-Indic numerals)
ARTICLE_PATTERN = r"(?=المادة\s+[0-9٠-٩]+)"
