"""Embed chunks and store in ChromaDB."""

from __future__ import annotations

from pathlib import Path

import chromadb
import structlog
from sentence_transformers import SentenceTransformer

from src.indexing.chunker import Chunk

log = structlog.get_logger()

_model_cache: dict[str, SentenceTransformer] = {}


def get_embedding_model(model_name: str = "BAAI/bge-small-en-v1.5") -> SentenceTransformer:
    """Load embedding model with caching.

    BGE-small chosen because:
    - Free, no API cost (runs locally)
    - 384 dimensions — small index, fast similarity search
    - Strong performance on MTEB retrieval benchmarks for its size
    - Good enough for a 200-500 paper corpus; a larger model (bge-large,
      OpenAI ada-002) would be overkill here and add cost
    """
    if model_name not in _model_cache:
        log.info("loading_embedding_model", model=model_name)
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


def build_index(
    chunks: list[Chunk],
    persist_dir: str = "data/index",
    collection_name: str = "papers",
    model_name: str = "BAAI/bge-small-en-v1.5",
) -> chromadb.Collection:
    """Embed all chunks and upsert into a ChromaDB collection.

    ChromaDB chosen because:
    - Runs locally with persistent storage (no server/hosting cost)
    - Built-in support for metadata filtering
    - Simple API — no YAML config or container setup like Qdrant
    - For 500 papers × ~5 chunks each ≈ 2500 vectors, even brute-force
      search is fast. We don't need HNSW tuning at this scale.
    """
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    model = get_embedding_model(model_name)
    batch_size = 64

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c.text for c in batch]
        embeddings = model.encode(texts, normalize_embeddings=True).tolist()

        collection.upsert(
            ids=[c.chunk_id for c in batch],
            embeddings=embeddings,
            documents=texts,
            metadatas=[
                {
                    "arxiv_id": c.arxiv_id,
                    "paper_title": c.paper_title,
                    "section_name": c.section_name,
                    "page_start": c.page_start,
                    "page_end": c.page_end,
                }
                for c in batch
            ],
        )
        log.info("indexed_batch", batch_start=i, batch_size=len(batch))

    log.info("index_complete", total_chunks=collection.count())
    return collection
