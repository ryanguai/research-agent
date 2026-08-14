"""LLM-as-judge for evaluating answer quality."""

from __future__ import annotations

import json
import os

import structlog

log = structlog.get_logger()

JUDGE_PROMPT = """You are evaluating an AI research assistant's answer to a question about ML papers.

Score the answer on a scale of 1-5:
5 = Correct, well-cited, covers all key points from the reference
4 = Mostly correct with minor omissions or imprecisions
3 = Partially correct but missing important information
2 = Contains some relevant information but mostly wrong or off-topic
1 = Completely wrong, hallucinated, or did not answer the question

Evaluation criteria:
- Does the answer match the reference answer's key claims?
- Are citations present and do they reference real papers?
- Is the answer grounded in the retrieved content (not hallucinated)?
- For adversarial questions: did it correctly decline to answer?

Respond with ONLY a JSON object: {"score": N, "reasoning": "brief explanation"}"""


def judge_answer(
    question: str,
    generated_answer: str,
    reference_answer: str,
    category: str,
    provider: str = "gemini",
    model: str | None = None,
) -> dict:
    """Score a generated answer against a reference using LLM-as-judge."""
    user_msg = f"""Question: {question}
Category: {category}

Generated Answer:
{generated_answer}

Reference Answer:
{reference_answer}

Score the generated answer (1-5). Respond with ONLY a JSON object."""

    try:
        if provider == "gemini":
            text = _judge_gemini(user_msg, model)
        else:
            text = _judge_openai_compat(provider, user_msg, model)

        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            result = json.loads(text[start:end])
            return {
                "score": float(result.get("score", 0)),
                "reasoning": result.get("reasoning", ""),
            }
    except Exception as e:
        log.warning("judge_failed", error=str(e))

    return {"score": 0.0, "reasoning": "Judge failed to produce a score"}


def _judge_gemini(user_msg: str, model: str | None) -> str:
    from dotenv import load_dotenv
    from google import genai
    from google.genai import types

    load_dotenv()
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    response = client.models.generate_content(
        model=model or "gemini-3.1-flash-lite",
        contents=[
            types.Content(
                role="user",
                parts=[types.Part(text=f"{JUDGE_PROMPT}\n\n{user_msg}")],
            )
        ],
        config=types.GenerateContentConfig(temperature=0.0),
    )
    return response.text


def _judge_openai_compat(provider: str, user_msg: str, model: str | None) -> str:
    from openai import OpenAI

    if provider == "ollama":
        client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        model = model or "llama3.2"
    else:
        client = OpenAI()
        model = model or "gpt-4o-mini"

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0,
    )
    return resp.choices[0].message.content
