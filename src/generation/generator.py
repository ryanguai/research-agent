"""Generate answers from retrieved chunks with citations."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from src.retrieval.vector_retriever import RetrievedChunk

log = structlog.get_logger()

SYSTEM_PROMPT = """You are a research assistant that answers questions about AI/ML research papers.

Rules:
1. Answer ONLY based on the provided paper excerpts. Never use your own knowledge.
2. CITATION FORMAT (mandatory): Every claim must cite the actual paper title and arXiv ID like this:
   According to "Paper Title" (arXiv: 2401.12345), Section: methods, ...
   NEVER use "[Source 1]" or "[Source N]" references. Always use the real paper title and arXiv ID from the excerpt header.
3. If the provided excerpts do not contain enough information to answer confidently, say:
   "I don't have enough information in the retrieved papers to answer this question confidently."
   Do NOT guess or fill in gaps with general knowledge.
4. DEDUPLICATION: If multiple excerpts describe the same idea, consolidate them into one point and cite all relevant papers together. Do not list the same concept multiple times with different wording.
5. BREADTH: Cover the full range of distinct approaches mentioned across ALL provided excerpts, not just the first few. Group related ideas under clear categories.
6. For synthesis questions comparing multiple papers, explicitly name each paper and what it contributes.
7. Be precise and technical. This is for researchers, not a general audience."""


def build_context(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks into a context block for the LLM."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[Source {i}] {chunk.paper_title} (arXiv: {chunk.arxiv_id}), "
            f"Section: {chunk.section_name}\n{chunk.text}"
        )
    return "\n\n---\n\n".join(parts)


@dataclass
class GenerationResult:
    answer: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float


def _get_env(key: str) -> str:
    from dotenv import load_dotenv
    load_dotenv()
    val = os.environ.get(key)
    if not val:
        raise ValueError(f"Set {key} in your .env file")
    return val


PROVIDER_DEFAULTS = {
    "ollama": "llama3.2",
    "openai": "gpt-4o-mini",
    "gemini": "gemini-3.1-flash-lite",
    "groq": "llama-3.1-8b-instant",
}


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=10, max=60))
def generate_answer(
    query: str,
    chunks: list[RetrievedChunk],
    provider: str = "ollama",
    model: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 1024,
) -> GenerationResult:
    """Generate a cited answer from retrieved chunks."""
    if model is None:
        model = PROVIDER_DEFAULTS[provider]

    context = build_context(chunks)
    user_message = f"""Based on the following paper excerpts, answer this question:

Question: {query}

Paper excerpts:
{context}"""

    start = time.time()

    if provider == "gemini":
        answer, input_tokens, output_tokens = _generate_gemini(
            model, user_message, temperature, max_tokens
        )
    else:
        answer, input_tokens, output_tokens = _generate_openai_compat(
            provider, model, user_message, temperature, max_tokens
        )

    latency_ms = (time.time() - start) * 1000
    cost = _estimate_cost(provider, model, input_tokens, output_tokens)

    result = GenerationResult(
        answer=answer,
        model=f"{provider}/{model}",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
        cost_usd=cost,
    )

    log.info(
        "generated_answer",
        provider=provider,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=round(latency_ms),
        cost_usd=round(cost, 5),
    )
    return result


def _generate_gemini(
    model: str, user_message: str, temperature: float, max_tokens: int
) -> tuple[str, int, int]:
    """Generate using Google's native GenAI SDK."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=_get_env("GEMINI_API_KEY"))
    response = client.models.generate_content(
        model=model,
        contents=[
            types.Content(
                role="user",
                parts=[types.Part(text=f"{SYSTEM_PROMPT}\n\n{user_message}")],
            )
        ],
        config=types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        ),
    )

    input_tokens = response.usage_metadata.prompt_token_count or 0
    output_tokens = response.usage_metadata.candidates_token_count or 0
    return response.text, input_tokens, output_tokens


def _generate_openai_compat(
    provider: str, model: str, user_message: str, temperature: float, max_tokens: int
) -> tuple[str, int, int]:
    """Generate using OpenAI-compatible API (Ollama, OpenAI, Groq)."""
    from openai import OpenAI

    if provider == "ollama":
        client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    elif provider == "groq":
        client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=_get_env("GROQ_API_KEY"),
        )
    else:
        client = OpenAI()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    usage = response.usage
    input_tokens = usage.prompt_tokens if usage else 0
    output_tokens = usage.completion_tokens if usage else 0
    return response.choices[0].message.content, input_tokens, output_tokens


def _estimate_cost(provider: str, model: str, input_tokens: int, output_tokens: int) -> float:
    """Rough cost estimate. Ollama and local models are free."""
    pricing = {
        "openai/gpt-4o-mini": (0.15 / 1_000_000, 0.60 / 1_000_000),
        "openai/gpt-4o": (2.50 / 1_000_000, 10.00 / 1_000_000),
        "gemini/gemini-flash-latest": (0.10 / 1_000_000, 0.40 / 1_000_000),
        "groq/llama-3.1-8b-instant": (0.05 / 1_000_000, 0.08 / 1_000_000),
    }
    key = f"{provider}/{model}"
    input_rate, output_rate = pricing.get(key, (0.0, 0.0))
    return input_tokens * input_rate + output_tokens * output_rate
