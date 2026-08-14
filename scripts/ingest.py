"""Script to run the full ingestion pipeline: fetch → parse → chunk → index."""

from __future__ import annotations

import json
from pathlib import Path

import structlog
import yaml

from src.ingestion.arxiv_client import PaperMeta, download_pdf, fetch_papers
from src.ingestion.pdf_parser import parse_pdf
from src.indexing.chunker import chunk_sections
from src.indexing.embedder import build_index

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ]
)
log = structlog.get_logger()


def main() -> None:
    with open("configs/defaults.yaml") as f:
        config = yaml.safe_load(f)

    corpus = config["corpus"]
    log.info("starting_ingestion", config=corpus)

    # Step 1: Fetch paper metadata from arXiv
    papers = fetch_papers(
        query=corpus["subtopic_query"],
        categories=corpus["categories"],
        max_results=corpus["max_results"],
    )

    # Save metadata for reproducibility
    meta_path = Path("data/raw/metadata.json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump([p.__dict__ for p in papers], f, indent=2, default=str)
    log.info("saved_metadata", path=str(meta_path), count=len(papers))

    # Step 2: Download PDFs
    pdf_dir = Path("data/raw/pdfs")
    for paper in papers:
        try:
            paper.pdf_path = download_pdf(paper, pdf_dir)
        except Exception as e:
            log.warning("pdf_download_failed", arxiv_id=paper.arxiv_id, error=str(e))

    papers_with_pdfs = [p for p in papers if p.pdf_path]
    log.info("pdfs_downloaded", total=len(papers_with_pdfs), failed=len(papers) - len(papers_with_pdfs))

    # Step 3: Parse PDFs into sections
    all_chunks = []
    for paper in papers_with_pdfs:
        try:
            sections = parse_pdf(str(paper.pdf_path))
            chunks = chunk_sections(
                arxiv_id=paper.arxiv_id,
                paper_title=paper.title,
                sections=sections,
                overlap_sentences=config["chunking"]["overlap_sentences"],
            )
            all_chunks.extend(chunks)
        except Exception as e:
            log.warning("parse_failed", arxiv_id=paper.arxiv_id, error=str(e))

    log.info("parsing_complete", total_chunks=len(all_chunks))

    # Step 4: Embed and index
    collection = build_index(
        chunks=all_chunks,
        persist_dir="data/index",
        model_name=config["embedding"]["model"],
    )
    log.info("indexing_complete", collection_count=collection.count())


if __name__ == "__main__":
    main()
