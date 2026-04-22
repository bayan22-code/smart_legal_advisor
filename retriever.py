"""
Hybrid retriever: vector search + BM25.
Merges and deduplicates results, returns top 10 candidate chunks.
"""

import re
from typing import Any

import chromadb
import numpy as np

from config import (
    CHROMA_COLLECTION_NAME,
    VECTOR_DB_PATH,
    TOP_K_HYBRID,
    TOP_K_VECTOR,
    TOP_K_BM25,
)
from embeddings import ArabicEmbedder
from bm25_search import BM25Index
from text_normalization import clean_text


class HybridRetriever:
    MIN_SIMILARITY_SCORE = 0.10

    """Combines ChromaDB vector search and BM25 keyword search."""

    def __init__(
        self,
        embedder: ArabicEmbedder,
        bm25_index: BM25Index,
        top_k_hybrid: int = TOP_K_HYBRID,
    ):
        self.embedder = embedder
        self.bm25_index = bm25_index
        self.top_k_hybrid = top_k_hybrid
        self.chroma_client = chromadb.PersistentClient(path=str(VECTOR_DB_PATH))
        self.collection = self.chroma_client.get_or_create_collection(CHROMA_COLLECTION_NAME)

    def _normalize_scores(self, results: list[tuple[dict, float]]) -> list[tuple[dict, float]]:
        """Min-max normalize scores to [0, 1] for fusion."""
        if not results:
            return []
        scores = [r[1] for r in results]
        lo, hi = min(scores), max(scores)
        if hi - lo == 0:
            return [(r[0], 1.0) for r in results]
        return [(r[0], (r[1] - lo) / (hi - lo)) for r in results]

    @staticmethod
    def _source_filter(database_route: str) -> dict[str, Any] | None:
        route_to_source = {
            "civil": "Civil Law",
            "penal": "Penal Law",
            "personal_status": "Personal Status Law",
            "all": None,
        }
        source = route_to_source.get(database_route, None)
        if source is None:
            return None
        return {"source": {"$eq": source}}

    def _extract_cross_references(self, chunks: list[dict[str, Any]]) -> list[int]:
        article_ids: set[int] = set()
        for chunk in chunks:
            text = chunk.get("text", "")
            for match in re.finditer(r"(?:المادة|Article)\s*(\d+)", text, flags=re.IGNORECASE):
                article_ids.add(int(match.group(1)))
        return sorted(article_ids)

    def fetch_by_article_numbers(
        self,
        article_numbers: list[int],
        database_route: str = "all",
    ) -> list[dict[str, Any]]:
        if not article_numbers:
            return []
        where_route = self._source_filter(database_route)
        results: list[dict[str, Any]] = []
        for n in article_numbers[:10]:
            where_article: dict[str, Any] = {"article_number": {"$eq": n}}
            where = {"$and": [where_route, where_article]} if where_route else where_article
            query = self.collection.get(
                where=where,
                include=["documents", "metadatas"],
                limit=2,
            )
            docs = query.get("documents", []) or []
            metas = query.get("metadatas", []) or []
            for i, doc in enumerate(docs):
                results.append({"text": doc, "metadata": metas[i] if i < len(metas) else {}})
        return results

    def retrieve(self, query: str | list[str], database_route: str = "all") -> list[dict[str, Any]]:
        """
        Hybrid retrieval: vector + BM25, merge, deduplicate, return top_k.
        """
        queries = query if isinstance(query, list) else [query]
        queries = [clean_text(q, apply_reversal_fix=False) for q in queries if q]
        if not queries:
            return []
        print(f"[Retriever] Searching Vector Store... route={database_route} variations={len(queries)}")

        where = self._source_filter(database_route)
        vector_chunks = []
        bm25_results = []
        for q in queries:
            q_emb = self.embedder.encode_query(q)
            emb_dim = len(q_emb.tolist()) if hasattr(q_emb, "tolist") else len(q_emb)
            print(f"[Retriever] Query embedding generated. dim={emb_dim}")
            vector_results = self.collection.query(
                query_embeddings=[q_emb.tolist()],
                n_results=4,
                include=["documents", "metadatas", "distances"],
                where=where,
            )
            if where and not (vector_results.get("documents") and vector_results["documents"][0]):
                vector_results = self.collection.query(
                    query_embeddings=[q_emb.tolist()],
                    n_results=4,
                    include=["documents", "metadatas", "distances"],
                )
            if vector_results["documents"] and vector_results["documents"][0]:
                for i, doc in enumerate(vector_results["documents"][0]):
                    dist = vector_results["distances"][0][i] if vector_results["distances"] else 0
                    # Similarity score for similarity_search_with_score style filtering.
                    score = 1.0 / (1.0 + dist)
                    meta = vector_results["metadatas"][0][i] if vector_results["metadatas"] else {}
                    if meta.get("article_number") is None or score < self.MIN_SIMILARITY_SCORE:
                        continue
                    vector_chunks.append(({"text": doc, "metadata": meta}, score))
            bm25_results.extend(self.bm25_index.search(q, top_k=2))

        vector_chunks = self._normalize_scores(vector_chunks)
        print(f"[Retriever] Vector hits={len(vector_chunks)}")

        # BM25 search
        if where:
            allowed_source = where["source"]["$eq"]
            bm25_results = [
                (chunk, score)
                for chunk, score in bm25_results
                if chunk.get("metadata", {}).get("source") == allowed_source
            ]
            if not bm25_results:
                bm25_results = []
                for q in queries:
                    bm25_results.extend(self.bm25_index.search(q, top_k=2))
        bm25_normalized = self._normalize_scores(bm25_results)
        print(f"[Retriever] BM25 hits={len(bm25_normalized)}")

        # Reciprocal rank fusion
        seen = {}
        for rank, (chunk, _) in enumerate(vector_chunks, start=1):
            key = chunk["text"][:200]  # Dedup key
            if key not in seen:
                seen[key] = {"chunk": chunk, "score": 0.0}
            seen[key]["score"] += 1.0 / rank

        for rank, (chunk, _) in enumerate(bm25_normalized, start=1):
            key = chunk["text"][:200]
            if key not in seen:
                seen[key] = {"chunk": chunk, "score": 0.0}
            seen[key]["score"] += 1.0 / rank

        # Sort by fused score, take top_k
        sorted_items = sorted(seen.values(), key=lambda x: x["score"], reverse=True)
        first_hop = [item["chunk"] for item in sorted_items[:2]]

        # Multi-hop retrieval: follow article cross-references mentioned in first hop chunks.
        referenced_article_ids = self._extract_cross_references(first_hop)
        second_hop = self.fetch_by_article_numbers(referenced_article_ids, database_route=database_route)

        merged = {chunk["text"][:220]: chunk for chunk in first_hop}
        for chunk in second_hop:
            merged.setdefault(chunk["text"][:220], chunk)
        merged_chunks = list(merged.values())
        final_results = self._mmr_select(
            query=queries[0],
            chunks=merged_chunks,
            k=2,
            lambda_param=0.7,
        )
        final_results = [c for c in final_results if c.get("metadata", {}).get("article_number") is not None][:2]

        # Safety net: if semantic path is empty, force lexical retrieval from BM25 index.
        if not final_results:
            fallback_pool: list[tuple[dict[str, Any], float]] = []
            for q in queries:
                fallback_pool.extend(self.bm25_index.search(q, top_k=8))
            if where:
                allowed_source = where["source"]["$eq"]
                fallback_pool = [
                    (chunk, score)
                    for chunk, score in fallback_pool
                    if chunk.get("metadata", {}).get("source") == allowed_source
                ]
            dedup: dict[str, dict[str, Any]] = {}
            for chunk, _score in sorted(fallback_pool, key=lambda item: item[1], reverse=True):
                key = chunk.get("text", "")[:220]
                if key and key not in dedup:
                    dedup[key] = chunk
                if len(dedup) >= 2:
                    break
            final_results = list(dedup.values())
        print(f"[Retriever] Found {len(final_results)} articles after merge.")
        return final_results

    def _mmr_select(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        k: int = 5,
        lambda_param: float = 0.7,
    ) -> list[dict[str, Any]]:
        """Select diverse-yet-relevant chunks with Maximal Marginal Relevance."""
        if not chunks:
            return []
        if len(chunks) <= k:
            return chunks

        texts = [c.get("text", "") for c in chunks]
        q_vec = self.embedder.encode_query(query)
        d_vecs = self.embedder.encode_passages(texts, show_progress=False)

        q = np.asarray(q_vec, dtype=float)
        docs = np.asarray(d_vecs, dtype=float)

        def _normalize(v: np.ndarray) -> np.ndarray:
            n = np.linalg.norm(v, axis=-1, keepdims=True)
            n[n == 0.0] = 1.0
            return v / n

        qn = _normalize(q.reshape(1, -1))[0]
        dn = _normalize(docs)
        relevance = dn @ qn

        selected: list[int] = [int(np.argmax(relevance))]
        candidates = set(range(len(chunks))) - set(selected)

        while candidates and len(selected) < k:
            best_idx = None
            best_score = -1e9
            for idx in candidates:
                similarity_to_selected = max(float(dn[idx] @ dn[s]) for s in selected)
                mmr_score = lambda_param * float(relevance[idx]) - (1 - lambda_param) * similarity_to_selected
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
            if best_idx is None:
                break
            selected.append(best_idx)
            candidates.remove(best_idx)

        return [chunks[i] for i in selected]
