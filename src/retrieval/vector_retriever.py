"""Baseline semantic retrieval using ChromaDB vector search."""

from __future__ import annotations

from dataclasses import dataclass

import chromadb
import structlog
from sentence_transformers import SentenceTransformer

from src.indexing.embedder import get_embedding_model

log = structlog.get_logger()


@dataclass
class RetrievedChunk:
    chunk_id: str
    arxiv_id: str
    paper_title: str
    section_name: str
    text: str
    score: float


def vector_search(
    query: str,
    collection: chromadb.Collection,
    model: SentenceTransformer | None = None,
    top_k: int = 10,
) -> list[RetrievedChunk]:
    """Pure vector (semantic) retrieval.

    Embeds the query with the same model used for indexing, then does
    cosine similarity search in ChromaDB. This is the baseline retrieval
    strategy we'll compare against hybrid in Phase 4.
    """
    if model is None:
        model = get_embedding_model()

    query_embedding = model.encode([query], normalize_embeddings=True).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        chunks.append(
            RetrievedChunk(
                chunk_id=results["ids"][0][i],
                arxiv_id=meta["arxiv_id"],
                paper_title=meta["paper_title"],
                section_name=meta["section_name"],
                text=results["documents"][0][i],
                score=1 - results["distances"][0][i],  # cosine distance → similarity
            )
        )

    log.info("vector_search", query=query[:80], num_results=len(chunks))
    return chunks
