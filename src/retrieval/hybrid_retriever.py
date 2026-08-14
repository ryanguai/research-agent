"""Hybrid retrieval: BM25 (keyword) + vector (semantic) with score fusion."""

from __future__ import annotations

import chromadb
import structlog
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.indexing.embedder import get_embedding_model
from src.retrieval.vector_retriever import RetrievedChunk, vector_search

log = structlog.get_logger()


class HybridRetriever:
    """Combines BM25 keyword search with vector semantic search.

    Why hybrid matters (interview talking point):
    - Vector search is great at semantic similarity ('machine learning' matches
      'neural network training') but can miss exact technical terms
    - BM25 is great at exact keyword matching ('LoRA', 'FlashAttention') but
      misses paraphrases
    - Combining both with Reciprocal Rank Fusion (RRF) gets the best of both
    - This is the comparison we run in Phase 4 to show engineering rigor
    """

    def __init__(
        self,
        collection: chromadb.Collection,
        model: SentenceTransformer | None = None,
        rrf_k: int = 60,
    ):
        self.collection = collection
        self.model = model or get_embedding_model()
        self.rrf_k = rrf_k

        all_docs = collection.get(include=["documents", "metadatas"])
        self.doc_ids = all_docs["ids"]
        self.doc_texts = all_docs["documents"]
        self.doc_metas = all_docs["metadatas"]

        tokenized = [doc.lower().split() for doc in self.doc_texts]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = 10) -> list[RetrievedChunk]:
        """Run both retrieval strategies and fuse with RRF."""
        vector_results = vector_search(query, self.collection, self.model, top_k=top_k * 2)
        bm25_results = self._bm25_search(query, top_k=top_k * 2)

        fused = self._reciprocal_rank_fusion(vector_results, bm25_results)
        final = sorted(fused.values(), key=lambda c: c.score, reverse=True)[:top_k]

        log.info("hybrid_search", query=query[:80], num_results=len(final))
        return final

    def _bm25_search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            meta = self.doc_metas[idx]
            results.append(
                RetrievedChunk(
                    chunk_id=self.doc_ids[idx],
                    arxiv_id=meta["arxiv_id"],
                    paper_title=meta["paper_title"],
                    section_name=meta["section_name"],
                    text=self.doc_texts[idx],
                    score=float(scores[idx]),
                )
            )
        return results

    def _reciprocal_rank_fusion(
        self,
        vector_results: list[RetrievedChunk],
        bm25_results: list[RetrievedChunk],
    ) -> dict[str, RetrievedChunk]:
        """Reciprocal Rank Fusion — merges two ranked lists.

        RRF score = sum(1 / (k + rank)) across all lists where the doc appears.
        k=60 is the standard default from the original RRF paper (Cormack et al. 2009).
        """
        fused: dict[str, RetrievedChunk] = {}

        for rank, chunk in enumerate(vector_results):
            rrf_score = 1.0 / (self.rrf_k + rank + 1)
            if chunk.chunk_id in fused:
                fused[chunk.chunk_id].score += rrf_score
            else:
                fused[chunk.chunk_id] = RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    arxiv_id=chunk.arxiv_id,
                    paper_title=chunk.paper_title,
                    section_name=chunk.section_name,
                    text=chunk.text,
                    score=rrf_score,
                )

        for rank, chunk in enumerate(bm25_results):
            rrf_score = 1.0 / (self.rrf_k + rank + 1)
            if chunk.chunk_id in fused:
                fused[chunk.chunk_id].score += rrf_score
            else:
                fused[chunk.chunk_id] = RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    arxiv_id=chunk.arxiv_id,
                    paper_title=chunk.paper_title,
                    section_name=chunk.section_name,
                    text=chunk.text,
                    score=rrf_score,
                )

        return fused
