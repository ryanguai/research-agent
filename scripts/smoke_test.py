"""Quick smoke test: fetch 5 papers, parse, chunk, index, query."""

import sys
sys.path.insert(0, "/Users/ryanguai/Projects/research-agent")

import structlog

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ]
)

from src.ingestion.arxiv_client import fetch_papers, download_pdf
from src.ingestion.pdf_parser import parse_pdf
from src.indexing.chunker import chunk_sections
from src.indexing.embedder import build_index
from src.retrieval.vector_retriever import vector_search
from pathlib import Path


def main():
    print("=== Step 1: Fetch papers from arXiv ===")
    papers = fetch_papers(
        query="RAG retrieval-augmented generation",
        categories=["cs.CL", "cs.LG"],
        max_results=5,
    )
    print(f"Fetched {len(papers)} papers\n")

    print("=== Step 2: Download PDFs ===")
    pdf_dir = Path("/Users/ryanguai/Projects/research-agent/data/raw/pdfs")
    for p in papers:
        try:
            p.pdf_path = download_pdf(p, pdf_dir)
        except Exception as e:
            print(f"  Failed: {p.arxiv_id} — {e}")
    papers = [p for p in papers if p.pdf_path]
    print(f"Downloaded {len(papers)} PDFs\n")

    print("=== Step 3: Parse PDFs into sections ===")
    all_chunks = []
    for p in papers:
        try:
            sections = parse_pdf(str(p.pdf_path))
            chunks = chunk_sections(p.arxiv_id, p.title, sections)
            all_chunks.extend(chunks)
            print(f"  {p.arxiv_id}: {len(sections)} sections → {len(chunks)} chunks")
        except Exception as e:
            print(f"  Failed: {p.arxiv_id} — {e}")
    print(f"Total chunks: {len(all_chunks)}\n")

    print("=== Step 4: Embed and index ===")
    index_dir = "/Users/ryanguai/Projects/research-agent/data/index_smoke"
    collection = build_index(all_chunks, persist_dir=index_dir, collection_name="smoke")
    print(f"Indexed {collection.count()} chunks\n")

    print("=== Step 5: Query ===")
    query = "What are the main challenges in retrieval-augmented generation?"
    results = vector_search(query, collection, top_k=3)
    for i, r in enumerate(results, 1):
        print(f"\n--- Result {i} (score: {r.score:.3f}) ---")
        print(f"Paper: {r.paper_title}")
        print(f"Section: {r.section_name}")
        print(f"Text: {r.text[:200]}...")

    print("\n=== Smoke test passed ===")


if __name__ == "__main__":
    main()
