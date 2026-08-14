"""Chunk parsed sections for embedding and indexing."""

from __future__ import annotations

from dataclasses import dataclass

import structlog

log = structlog.get_logger()


@dataclass
class Chunk:
    chunk_id: str
    arxiv_id: str
    paper_title: str
    section_name: str
    text: str
    page_start: int
    page_end: int


def chunk_sections(
    arxiv_id: str,
    paper_title: str,
    sections: list,
    overlap_sentences: int = 2,
    max_chunk_chars: int = 1500,
) -> list[Chunk]:
    """Split sections into chunks, respecting section boundaries.

    Strategy: chunk by section first (each section is one chunk). If a section
    is too long, split it at sentence boundaries with overlap. This preserves
    semantic coherence better than fixed-window chunking.

    Why section-aware chunking matters (interview talking point):
    - Fixed-window chunking can split a method description mid-sentence,
      putting the setup in one chunk and the key result in another
    - Section-aware chunking keeps 'methods' together, 'results' together
    - The overlap at boundaries catches cases where a conclusion references
      a result from the previous paragraph
    """
    chunks = []
    section_counts: dict[str, int] = {}
    for section in sections:
        sec_idx = section_counts.get(section.name, 0)
        section_counts[section.name] = sec_idx + 1
        id_prefix = f"{arxiv_id}::{section.name}_{sec_idx}"

        if len(section.text) <= max_chunk_chars:
            chunks.append(
                Chunk(
                    chunk_id=f"{id_prefix}::0",
                    arxiv_id=arxiv_id,
                    paper_title=paper_title,
                    section_name=section.name,
                    text=section.text,
                    page_start=section.page_start,
                    page_end=section.page_end,
                )
            )
        else:
            sub_chunks = _split_with_overlap(section.text, max_chunk_chars, overlap_sentences)
            for i, text in enumerate(sub_chunks):
                chunks.append(
                    Chunk(
                        chunk_id=f"{id_prefix}::{i}",
                        arxiv_id=arxiv_id,
                        paper_title=paper_title,
                        section_name=section.name,
                        text=text,
                        page_start=section.page_start,
                        page_end=section.page_end,
                    )
                )

    log.info("chunked_paper", arxiv_id=arxiv_id, num_chunks=len(chunks))
    return chunks


def _split_with_overlap(
    text: str, max_chars: int, overlap_sentences: int
) -> list[str]:
    """Split text into chunks at sentence boundaries with overlap."""
    sentences = _split_sentences(text)
    chunks = []
    start = 0

    while start < len(sentences):
        current = []
        char_count = 0

        for i in range(start, len(sentences)):
            if char_count + len(sentences[i]) > max_chars and current:
                break
            current.append(sentences[i])
            char_count += len(sentences[i])

        chunks.append(" ".join(current))
        advance = len(current) - overlap_sentences
        if advance < 1:
            advance = max(len(current), 1)
        start += advance

    return chunks


def _split_sentences(text: str) -> list[str]:
    """Naive sentence splitter — good enough for academic text."""
    import re

    raw = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in raw if s.strip()]
