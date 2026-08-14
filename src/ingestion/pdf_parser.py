"""Parse PDF text into structured sections."""

from __future__ import annotations

import re
from dataclasses import dataclass

import fitz  # PyMuPDF
import structlog

log = structlog.get_logger()

SECTION_PATTERNS = [
    r"^\s*(?:\d+\.?\s+)?(abstract)\s*$",
    r"^\s*(?:\d+\.?\s+)?(introduction)\s*$",
    r"^\s*(?:\d+\.?\s+)?(related\s+work)\s*$",
    r"^\s*(?:\d+\.?\s+)?(background)\s*$",
    r"^\s*(?:\d+\.?\s+)?(method(?:s|ology)?)\s*$",
    r"^\s*(?:\d+\.?\s+)?(approach)\s*$",
    r"^\s*(?:\d+\.?\s+)?(experiment(?:s|al)?(?:\s+(?:setup|results))?)\s*$",
    r"^\s*(?:\d+\.?\s+)?(results?(?:\s+and\s+discussion)?)\s*$",
    r"^\s*(?:\d+\.?\s+)?(discussion)\s*$",
    r"^\s*(?:\d+\.?\s+)?(conclusion(?:s)?)\s*$",
    r"^\s*(?:\d+\.?\s+)?(limitations?)\s*$",
    r"^\s*(?:\d+\.?\s+)?(references)\s*$",
    r"^\s*(?:\d+\.?\s+)?(appendi(?:x|ces))\s*$",
]

COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in SECTION_PATTERNS]


@dataclass
class Section:
    name: str
    text: str
    page_start: int
    page_end: int


def parse_pdf(pdf_path: str) -> list[Section]:
    """Extract text from a PDF and split into named sections.

    Strategy: extract full text page by page, then scan for lines matching
    common academic section headings. Text between headings becomes a section.
    Lines before the first heading go into a 'preamble' section.

    This is heuristic — academic PDFs have no universal structure. But it works
    well enough for arXiv papers, which mostly follow a standard layout.
    """
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    log.info("parsing_started", path=pdf_path, total_pages=total_pages)
    pages: list[tuple[int, str]] = []
    for page_num in range(total_pages):
        if (page_num + 1) % 10 == 0 or page_num == 0:
            log.debug("parsing_page", page=page_num + 1, of=total_pages)
        text = doc[page_num].get_text()
        pages.append((page_num + 1, text))
    doc.close()

    full_text_with_pages: list[tuple[int, str]] = []
    for page_num, text in pages:
        for line in text.split("\n"):
            full_text_with_pages.append((page_num, line))

    sections: list[Section] = []
    current_name = "preamble"
    current_lines: list[str] = []
    current_page_start = 1
    current_page_end = 1

    for page_num, line in full_text_with_pages:
        matched_section = None
        for pattern in COMPILED_PATTERNS:
            m = pattern.match(line.strip())
            if m:
                matched_section = m.group(1).lower().strip()
                break

        if matched_section:
            if current_lines:
                sections.append(
                    Section(
                        name=current_name,
                        text="\n".join(current_lines).strip(),
                        page_start=current_page_start,
                        page_end=current_page_end,
                    )
                )
            current_name = matched_section
            current_lines = []
            current_page_start = page_num
            current_page_end = page_num
        else:
            current_lines.append(line)
            current_page_end = page_num

    if current_lines:
        sections.append(
            Section(
                name=current_name,
                text="\n".join(current_lines).strip(),
                page_start=current_page_start,
                page_end=current_page_end,
            )
        )

    sections = [s for s in sections if s.name != "references" and len(s.text) > 50]

    log.info(
        "parsed_pdf",
        path=pdf_path,
        sections=[s.name for s in sections],
        total_chars=sum(len(s.text) for s in sections),
    )
    return sections
