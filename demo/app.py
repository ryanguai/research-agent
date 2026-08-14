"""Streamlit demo for the Research Paper Q&A Agent."""

import os
import subprocess
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

INDEX_DIR = Path("data/index")
INDEX_URL = "https://github.com/ryanguai/research-agent/releases/download/v0.1.0/research-agent-index.tar.gz"


@st.cache_resource
def ensure_index():
    """Download and extract the pre-built index if not present."""
    if INDEX_DIR.exists() and any(INDEX_DIR.iterdir()):
        return
    INDEX_DIR.parent.mkdir(parents=True, exist_ok=True)
    tar_path = Path("data/index.tar.gz")
    subprocess.run(["curl", "-L", "-o", str(tar_path), INDEX_URL], check=True)
    with tarfile.open(tar_path) as tar:
        tar.extractall("data")
    tar_path.unlink()


ensure_index()

try:
    gemini_key = st.secrets["GEMINI_API_KEY"]
    os.environ["GEMINI_API_KEY"] = gemini_key
    print(f"[app] Loaded GEMINI_API_KEY from st.secrets (length={len(gemini_key)})")
except Exception as e:
    print(f"[app] st.secrets failed: {e}, falling back to .env")
    from dotenv import load_dotenv
    load_dotenv()

print(f"[app] GEMINI_API_KEY in env: {bool(os.environ.get('GEMINI_API_KEY'))}")

from src.pipeline import Pipeline

st.set_page_config(page_title="Research Paper Q&A", page_icon="📄", layout="wide")

st.title("📄 AI Research Paper Q&A Agent")
st.markdown(
    "Ask questions about recent RAG (Retrieval-Augmented Generation) research papers. "
    "Answers are grounded in a corpus of ~500 arXiv papers from cs.CL and cs.LG."
)


@st.cache_resource
def load_pipeline(mode: str, provider: str) -> Pipeline:
    return Pipeline(
        index_dir=str(INDEX_DIR),
        retrieval_mode=mode,
        provider=provider,
        generation_model="gemini-3.1-flash-lite",
    )


# Sidebar config
with st.sidebar:
    st.header("Settings")
    retrieval_mode = st.selectbox("Retrieval Strategy", ["hybrid", "vector"], index=0)
    provider = "gemini"
    top_k = st.slider("Retrieved chunks (top-k)", 3, 20, 10)

    st.markdown("---")
    st.markdown("### About")
    st.markdown(
        "This system retrieves relevant sections from arXiv papers and generates "
        "cited answers. It explicitly declines when the corpus doesn't support "
        "a confident answer."
    )
    st.markdown(
        "**Retrieval strategies:**\n"
        "- **Vector**: Semantic similarity search (BGE-small embeddings)\n"
        "- **Hybrid**: BM25 keyword search + vector search, fused with RRF"
    )

pipeline = load_pipeline(retrieval_mode, provider)
pipeline.top_k = top_k

# Example questions
examples = [
    "What are the main approaches to improving retrieval quality in RAG systems?",
    "How does knowledge poisoning affect RAG systems and what defenses exist?",
    "What caching strategies have been proposed for efficient RAG serving?",
    "Compare different approaches to adaptive retrieval in RAG systems.",
]

st.markdown("**Try an example:**")
cols = st.columns(2)
for i, ex in enumerate(examples):
    if cols[i % 2].button(ex, key=f"ex_{i}"):
        st.session_state["query"] = ex

query = st.text_input(
    "Your question:",
    value=st.session_state.get("query", ""),
    placeholder="e.g., What chunking strategies work best for RAG?",
)

if query:
    with st.spinner("Retrieving and generating..."):
        try:
            answer, chunks, result = pipeline.query(query)
        except Exception as e:
            st.error("The LLM provider is temporarily rate-limited. Please wait a moment and try again.")
            st.stop()

    st.markdown("### Answer")
    st.markdown(answer)

    # Stats row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Latency", f"{result.latency_ms:.0f}ms")
    col2.metric("Cost", f"${result.cost_usd:.4f}")
    col3.metric("Chunks used", len(chunks))
    col4.metric("Papers cited", len(set(c.arxiv_id for c in chunks)))

    # Retrieved sources
    with st.expander(f"📚 Retrieved Sources ({len(chunks)} chunks from {len(set(c.arxiv_id for c in chunks))} papers)"):
        for i, chunk in enumerate(chunks):
            st.markdown(f"**[{i+1}] {chunk.paper_title}** (arXiv: {chunk.arxiv_id})")
            st.markdown(f"*Section: {chunk.section_name} | Score: {chunk.score:.3f}*")
            st.text(chunk.text[:500] + ("..." if len(chunk.text) > 500 else ""))
            st.markdown("---")
