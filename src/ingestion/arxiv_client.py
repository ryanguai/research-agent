"""Fetch paper metadata and PDFs from the arXiv API."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import arxiv
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

log = structlog.get_logger()


@dataclass
class PaperMeta:
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    published: str
    categories: list[str]
    pdf_url: str
    pdf_path: Path | None = None


def fetch_papers(
    query: str,
    categories: list[str],
    max_results: int = 500,
) -> list[PaperMeta]:
    """Search arXiv and return metadata for matching papers.

    Uses the `arxiv` Python client which wraps the Atom feed API.
    We filter by category in the query string itself (arXiv API supports
    `cat:cs.CL` syntax) so the server does the filtering, not us.
    """
    cat_filter = " OR ".join(f"cat:{c}" for c in categories)
    full_query = f"({query}) AND ({cat_filter})"

    client = arxiv.Client(page_size=100, delay_seconds=3.0, num_retries=3)
    search = arxiv.Search(
        query=full_query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    papers = []
    for result in client.results(search):
        papers.append(
            PaperMeta(
                arxiv_id=result.entry_id.split("/")[-1],
                title=result.title,
                authors=[a.name for a in result.authors],
                abstract=result.summary,
                published=result.published.isoformat(),
                categories=result.categories,
                pdf_url=result.pdf_url,
            )
        )
        log.info("fetched_paper", arxiv_id=papers[-1].arxiv_id, title=papers[-1].title)

    log.info("fetch_complete", total=len(papers))
    return papers


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
def download_pdf(paper: PaperMeta, output_dir: Path) -> Path:
    """Download a single PDF. Retries on transient failures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / f"{paper.arxiv_id.replace('/', '_')}.pdf"

    if dest.exists():
        log.debug("pdf_cached", arxiv_id=paper.arxiv_id)
        return dest

    import httpx

    log.info("downloading_pdf", arxiv_id=paper.arxiv_id)
    with httpx.Client(follow_redirects=True, timeout=60) as client:
        resp = client.get(paper.pdf_url)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
    time.sleep(1)  # respect arXiv rate limits
    return dest
