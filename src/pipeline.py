"""End-to-end pipeline: query → retrieve → generate."""

from __future__ import annotations

import chromadb
import structlog

from src.generation.generator import GenerationResult, generate_answer
from src.indexing.embedder import get_embedding_model
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.vector_retriever import RetrievedChunk, vector_search

log = structlog.get_logger()


class Pipeline:
    """Orchestrates retrieval and generation for a single query."""

    def __init__(
        self,
        index_dir: str = "data/index",
        collection_name: str = "papers",
        retrieval_mode: str = "vector",
        top_k: int = 10,
        provider: str = "ollama",
        generation_model: str | None = None,
    ):
        self.client = chromadb.PersistentClient(path=index_dir)
        self.collection = self.client.get_collection(collection_name)
        self.embedding_model = get_embedding_model()
        self.retrieval_mode = retrieval_mode
        self.top_k = top_k
        self.provider = provider
        self.generation_model = generation_model

        if retrieval_mode == "hybrid":
            self.hybrid = HybridRetriever(self.collection, self.embedding_model)

    def query(
        self, question: str
    ) -> tuple[str, list[RetrievedChunk], GenerationResult]:
        """Run the full pipeline: retrieve → generate."""
        if self.retrieval_mode == "hybrid":
            chunks = self.hybrid.search(question, top_k=self.top_k)
        else:
            chunks = vector_search(
                question, self.collection, self.embedding_model, top_k=self.top_k
            )

        result = generate_answer(
            question, chunks, provider=self.provider, model=self.generation_model
        )

        log.info(
            "pipeline_complete",
            mode=self.retrieval_mode,
            num_chunks=len(chunks),
            latency_ms=round(result.latency_ms),
        )

        return result.answer, chunks, result
