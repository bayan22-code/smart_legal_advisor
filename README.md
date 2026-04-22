# المستشار القانوني الذكي | Smart Legal Advisor

Production-grade Arabic Legal RAG system for the Syrian Civil Law.

## Features

- PDF Arabic text extraction (pypdf / pdfplumber)
- Smart legal article chunking by `المادة N` pattern
- Metadata extraction (article number, law, source)
- Embedding search (SentenceTransformers, multilingual-e5-large)
- BM25 keyword search (rank_bm25)
- Hybrid retrieval (vector + BM25, merged & deduplicated)
- Cross-encoder reranking (top 3 articles)
- RAG pipeline with legal citations
- OpenRouter LLM integration
- Streamlit ChatGPT-style interface

## Setup

```bash
cd smart_legaladvisor
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your OpenRouter API key:

```
OPENROUTER_API_KEY=your_api_key_here
```

Place `civil_law.pdf` in `legal_rag/data/`.

## Ingestion

```bash
cd legal_rag
python ingest.py
```

## Run the App

```bash
cd legal_rag
streamlit run streamlit_app.py
```

## Project Structure

```
legal_rag/
├── data/
│   └── civil_law.pdf
├── vector_db/           # ChromaDB (created by ingest)
├── config.py
├── pdf_loader.py
├── chunker.py
├── embeddings.py
├── bm25_search.py
├── retriever.py
├── reranker.py
├── rag_pipeline.py
├── openrouter_client.py
├── ingest.py
└── streamlit_app.py
```
