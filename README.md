# AI Research Paper Q&A Agent

A RAG (Retrieval-Augmented Generation) system that answers technical questions about recent AI/ML research papers, citing the specific paper and section it drew from. Built as a portfolio project demonstrating retrieval engineering, evaluation rigor, and production hardening.

## Problem Statement

Keeping up with AI/ML research is hard — hundreds of papers are published weekly on arXiv. This system lets you ask natural language questions about a curated corpus of ~500 RAG-related papers and get cited, grounded answers. It explicitly declines to answer when the corpus doesn't support a confident response, rather than hallucinating.

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   arXiv API  │────▶│  PDF Parser  │────▶│   Chunker    │────▶│  Embedder    │
│  (500 papers)│     │  (PyMuPDF)   │     │ (by section) │     │ (BGE-small)  │
└──────────────┘     └──────────────┘     └──────────────┘     └──────┬───────┘
                                                                      │
                                                                      ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Answer +   │◀────│  Generator   │◀────│  Retriever   │◀────│  ChromaDB    │
│  Citations   │     │ (Ollama/API) │     │(Vector/Hybrid│     │ (28K chunks) │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
```

**Ingestion**: Papers are fetched from arXiv (cs.CL + cs.LG, filtered for RAG-related work), PDFs are parsed into structured sections (abstract, methods, results, etc.) using PyMuPDF, chunked at section boundaries with sentence-level overlap, and embedded with BGE-small-en-v1.5 into ChromaDB.

**Retrieval**: Two strategies implemented for comparison:
- **Vector-only**: Cosine similarity search on BGE embeddings
- **Hybrid**: BM25 keyword search + vector search, fused with Reciprocal Rank Fusion (RRF)

**Generation**: Retrieved chunks are passed to an LLM with instructions to cite sources and decline when uncertain. Supports Ollama (local/free), OpenAI, Google Gemini, and Groq as providers.

## Design Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Sub-topic | RAG systems | Meta (building RAG about RAG), personally verifiable, active research area |
| Embedding model | BGE-small-en-v1.5 | Free (local), 384 dims, strong MTEB retrieval scores for its size. No API cost. Overkill models unnecessary at 2,500 chunks |
| Vector store | ChromaDB | Local, no hosting cost, built-in metadata filtering. FAISS is faster at scale but unnecessary for <30K vectors |
| Chunking | Section-aware | Preserves semantic coherence (methods stay with methods). Fixed-window chunking splits key findings mid-sentence |
| Hybrid retrieval | BM25 + Vector via RRF | Vector catches semantic similarity; BM25 catches exact terms (LoRA, FlashAttention). RRF fusion (k=60) from Cormack et al. 2009 |
| Generation LLM | Ollama (local) / Gemini Flash (demo) | Zero cost for development; free-tier API for deployed demo |

## Eval Results

84 questions (54 factual, 20 synthesis, 10 adversarial), each run against both retrieval strategies with Gemini 3.1 Flash Lite as generator and LLM-as-judge.

| Metric | Vector-Only | Hybrid | Delta |
|--------|------------|--------|-------|
| Overall Judge Score (1-5) | 2.88 | 2.81 | -0.07 |
| Factual Retrieval Recall | 57.4% | 68.5% | **+11.1%** |
| Factual Judge Score | 2.46 | 2.52 | +0.06 |
| Synthesis Judge Score | 2.95 | 3.35 | **+0.40** |
| Adversarial Decline Rate | 100% | 60% | -40% |
| Avg Latency (ms) | 4,630 | 5,816 | +1,186 |
| Total Cost ($) | 0.00 | 0.00 | 0.00 |

## Retrieval Comparison

The eval suite was run against both retrieval strategies to quantify the impact of adding BM25 keyword search to the vector baseline.

**Key findings:**

- **Hybrid retrieval improves recall by 11%** — BM25 catches papers with exact technical terms that vector search misses, bringing retrieval recall from 57.4% to 68.5%.
- **Synthesis questions benefit most from hybrid** — judge score improved from 2.95 to 3.35 (+0.40). Combining info across papers requires diverse retrieval, which hybrid delivers.
- **Vector-only is better at declining adversarial questions** — 100% decline rate vs 60%. Hybrid's broader retrieval pulls in loosely related chunks, making the generator less confident about saying "I don't know."
- **Hybrid adds ~1.2s latency** — the BM25 search over 28K chunks is the cost. Acceptable for a research Q&A tool, but worth noting.
- **Overall judge scores are close** — the Gemini 3.1 Flash Lite judge isn't highly discriminating. The retrieval metrics tell the clearer story.

## Project Structure

```
src/
  ingestion/       Fetch papers from arXiv, parse PDFs into sections
  indexing/        Chunk sections, embed with BGE-small, store in ChromaDB
  retrieval/       Vector search (baseline) + hybrid BM25+vector (comparison)
  generation/      LLM answer generation with citation instructions
  eval/            Evaluation runner, metrics, LLM-as-judge
  pipeline.py      Wires retrieval → generation into one call
scripts/           Ingestion, eval runner, load test, Q&A generation
demo/              Streamlit interactive demo
eval/              Test sets, raw results, comparison reports
configs/           YAML configuration (corpus, retrieval, generation params)
```

## Quick Start

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Ingest papers (takes ~30 min)
python -m scripts.ingest

# Run the demo
streamlit run demo/app.py

# Run eval suite
python scripts/generate_eval_candidates.py
python scripts/run_eval.py

# Load test
python scripts/load_test.py
```

## Tech Stack

- **Retrieval**: ChromaDB, BGE-small-en-v1.5, BM25 (rank-bm25)
- **Generation**: Ollama (Llama 3.2) / OpenAI / Google Gemini / Groq
- **PDF Parsing**: PyMuPDF (section-aware extraction)
- **Evaluation**: Custom eval runner + LLM-as-judge
- **Demo**: Streamlit
- **Logging**: structlog (structured, per-request)
