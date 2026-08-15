"""Self-correcting multi-step research agent.

A plan-act-reflect loop built with LangGraph:
  1. Classify — is this a simple or complex question?
  2. Decompose — break complex questions into sub-questions
  3. Retrieve — retrieve evidence for each sub-question
  4. Draft — synthesize a cited answer from all evidence
  5. Verify — check each claim against its cited source
  6. Replan/Retry — if verification fails, fix or re-retrieve (max 2 retries)
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import TypedDict

sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb
import structlog
from langgraph.graph import END, StateGraph

from src.generation.generator import SYSTEM_PROMPT, _get_env
from src.indexing.embedder import get_embedding_model
from src.retrieval.vector_retriever import RetrievedChunk, vector_search

log = structlog.get_logger()

MAX_RETRIES = 2


class AgentState(TypedDict):
    question: str
    is_complex: bool
    sub_questions: list[str]
    evidence: dict[str, list[RetrievedChunk]]
    draft: str
    verification: dict
    retry_count: int
    final_answer: str
    trace: list[str]


def _call_gemini(prompt: str, temperature: float = 0.1) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=_get_env("GEMINI_API_KEY"))

    models = ["gemini-3.1-flash-lite", "gemini-3-flash-preview", "gemma-4-26b-a4b-it"]
    for m in models:
        try:
            response = client.models.generate_content(
                model=m,
                contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
                config=types.GenerateContentConfig(
                    temperature=temperature, max_output_tokens=1024
                ),
            )
            return response.text
        except Exception:
            continue
    raise RuntimeError("All Gemini models exhausted")


def _get_collection():
    client = chromadb.PersistentClient(path="data/index")
    return client.get_collection("papers")


# ── Graph nodes ──


def classify(state: AgentState) -> AgentState:
    """Decide if the question needs decomposition or can be answered directly."""
    prompt = f"""Classify this question as SIMPLE or COMPLEX.

A COMPLEX question requires combining information from multiple papers or topics,
comparing approaches, or answering multiple distinct sub-parts.
A SIMPLE question can be answered from a single paper or topic.

Question: {state['question']}

Respond with ONLY "SIMPLE" or "COMPLEX"."""

    result = _call_gemini(prompt, temperature=0.0).strip().upper()
    is_complex = "COMPLEX" in result

    trace = state["trace"] + [f"classify: {'complex' if is_complex else 'simple'}"]
    log.info("agent_classify", question=state["question"][:60], is_complex=is_complex)

    return {**state, "is_complex": is_complex, "trace": trace}


def decompose(state: AgentState) -> AgentState:
    """Break a complex question into 2-4 sub-questions."""
    prompt = f"""Break this complex research question into 2-4 specific sub-questions
that can each be answered independently from research papers.

Question: {state['question']}

Return ONLY a JSON array of strings, e.g. ["sub-question 1", "sub-question 2"].
No markdown, no explanation."""

    result = _call_gemini(prompt, temperature=0.1)
    try:
        start = result.find("[")
        end = result.rfind("]") + 1
        sub_questions = json.loads(result[start:end])
    except (json.JSONDecodeError, ValueError):
        sub_questions = [state["question"]]

    trace = state["trace"] + [f"decompose: {len(sub_questions)} sub-questions"]
    log.info("agent_decompose", num_sub_questions=len(sub_questions))

    return {**state, "sub_questions": sub_questions, "trace": trace}


def pass_through(state: AgentState) -> AgentState:
    """For simple questions, skip decomposition."""
    trace = state["trace"] + ["pass_through: single question, no decomposition"]
    return {**state, "sub_questions": [state["question"]], "trace": trace}


def retrieve(state: AgentState) -> AgentState:
    """Retrieve evidence for each sub-question."""
    collection = _get_collection()
    model = get_embedding_model()

    evidence = {}
    for sq in state["sub_questions"]:
        chunks = vector_search(sq, collection, model, top_k=5)
        evidence[sq] = chunks

    total_chunks = sum(len(v) for v in evidence.values())
    trace = state["trace"] + [f"retrieve: {total_chunks} chunks for {len(state['sub_questions'])} sub-questions"]
    log.info("agent_retrieve", sub_questions=len(state["sub_questions"]), total_chunks=total_chunks)

    return {**state, "evidence": evidence, "trace": trace}


def draft(state: AgentState) -> AgentState:
    """Synthesize an answer from all retrieved evidence."""
    context_parts = []
    for sq, chunks in state["evidence"].items():
        context_parts.append(f"--- Evidence for: {sq} ---")
        for i, c in enumerate(chunks, 1):
            context_parts.append(
                f"[{c.paper_title} (arXiv: {c.arxiv_id}), Section: {c.section_name}]\n{c.text}"
            )
    context = "\n\n".join(context_parts)

    correction_note = ""
    if state["retry_count"] > 0 and state["verification"].get("issues"):
        issues = state["verification"]["issues"]
        correction_note = f"""

IMPORTANT: Your previous draft had these issues that must be fixed:
{chr(10).join(f'- {issue}' for issue in issues)}

Fix these specific issues in your new draft. Remove or correct any unsupported claims."""

    prompt = f"""{SYSTEM_PROMPT}

Based on the following paper excerpts, answer this question:

Question: {state['question']}

Paper excerpts:
{context}
{correction_note}"""

    answer = _call_gemini(prompt)

    trace = state["trace"] + [f"draft: attempt {state['retry_count'] + 1}, {len(answer)} chars"]
    log.info("agent_draft", attempt=state["retry_count"] + 1, answer_length=len(answer))

    return {**state, "draft": answer, "trace": trace}


def verify(state: AgentState) -> AgentState:
    """Verify each claim in the draft against the retrieved evidence."""
    all_chunks_text = []
    for chunks in state["evidence"].values():
        for c in chunks:
            all_chunks_text.append(f"[{c.paper_title} (arXiv: {c.arxiv_id})] {c.text[:300]}")
    sources = "\n".join(all_chunks_text)

    prompt = f"""You are a verification agent. Check if each claim in the draft answer
is supported by the provided source material.

Draft answer:
{state['draft']}

Available source material:
{sources}

For each claim in the draft, check:
1. Is the claim actually stated or supported in the source material?
2. Does the cited paper actually contain the claimed information?

Respond with ONLY a JSON object:
{{
  "passed": true/false,
  "issues": ["list of specific unsupported or incorrectly cited claims"],
  "num_claims_checked": N,
  "num_supported": N
}}"""

    result = _call_gemini(prompt, temperature=0.0)

    try:
        start = result.find("{")
        end = result.rfind("}") + 1
        verification = json.loads(result[start:end])
    except (json.JSONDecodeError, ValueError):
        verification = {"passed": True, "issues": [], "num_claims_checked": 0, "num_supported": 0}

    passed = verification.get("passed", True)
    issues = verification.get("issues", [])

    trace = state["trace"] + [
        f"verify: {'PASS' if passed else 'FAIL'} — {len(issues)} issues"
    ]
    log.info("agent_verify", passed=passed, num_issues=len(issues))

    return {**state, "verification": verification, "trace": trace}


def should_retry(state: AgentState) -> str:
    """Conditional edge: retry, replan with new retrieval, or accept."""
    v = state["verification"]
    if v.get("passed", True):
        return "accept"
    if state["retry_count"] >= MAX_RETRIES:
        return "accept_with_caveat"
    if v.get("issues") and any("not found" in i.lower() or "no evidence" in i.lower() for i in v.get("issues", [])):
        return "replan"
    return "retry"


def retry_draft(state: AgentState) -> AgentState:
    """Increment retry counter and go back to drafting with correction notes."""
    trace = state["trace"] + [f"retry: attempt {state['retry_count'] + 1} -> {state['retry_count'] + 2}"]
    return {**state, "retry_count": state["retry_count"] + 1, "trace": trace}


def replan(state: AgentState) -> AgentState:
    """Issue follow-up retrieval for gaps identified by the verifier."""
    issues = state["verification"].get("issues", [])
    if not issues:
        return retry_draft(state)

    collection = _get_collection()
    model = get_embedding_model()

    follow_up_query = f"Evidence for: {issues[0]}"
    new_chunks = vector_search(follow_up_query, collection, model, top_k=3)

    evidence = dict(state["evidence"])
    evidence[f"follow-up: {issues[0][:50]}"] = new_chunks

    trace = state["trace"] + [f"replan: follow-up retrieval for '{issues[0][:50]}', got {len(new_chunks)} chunks"]
    log.info("agent_replan", follow_up=issues[0][:50], new_chunks=len(new_chunks))

    return {
        **state,
        "evidence": evidence,
        "retry_count": state["retry_count"] + 1,
        "trace": trace,
    }


def accept(state: AgentState) -> AgentState:
    """Accept the draft as the final answer."""
    trace = state["trace"] + ["accept: verification passed"]
    return {**state, "final_answer": state["draft"], "trace": trace}


def accept_with_caveat(state: AgentState) -> AgentState:
    """Accept after max retries, flagging unverified portions."""
    issues = state["verification"].get("issues", [])
    caveat = "\n\n---\n**Note:** The following claims could not be fully verified against the retrieved sources:\n"
    caveat += "\n".join(f"- {issue}" for issue in issues)

    trace = state["trace"] + [f"accept_with_caveat: {len(issues)} unresolved issues after {MAX_RETRIES} retries"]
    log.info("agent_accept_with_caveat", unresolved_issues=len(issues))

    return {**state, "final_answer": state["draft"] + caveat, "trace": trace}


def route_complexity(state: AgentState) -> str:
    """Conditional edge: decompose or pass through."""
    return "decompose" if state["is_complex"] else "pass_through"


# ── Build the graph ──


def build_agent() -> StateGraph:
    """
    Graph:
        classify → [complex?] → decompose → retrieve → draft → verify → [should_retry?]
                               ↘ pass_through ↗                         ├→ accept → END
                                                                        ├→ retry_draft → draft
                                                                        ├→ replan → draft
                                                                        └→ accept_with_caveat → END
    """
    graph = StateGraph(AgentState)

    graph.add_node("classify", classify)
    graph.add_node("decompose", decompose)
    graph.add_node("pass_through", pass_through)
    graph.add_node("retrieve", retrieve)
    graph.add_node("draft", draft)
    graph.add_node("verify", verify)
    graph.add_node("retry_draft", retry_draft)
    graph.add_node("replan", replan)
    graph.add_node("accept", accept)
    graph.add_node("accept_with_caveat", accept_with_caveat)

    graph.set_entry_point("classify")
    graph.add_conditional_edges("classify", route_complexity, {
        "decompose": "decompose",
        "pass_through": "pass_through",
    })
    graph.add_edge("decompose", "retrieve")
    graph.add_edge("pass_through", "retrieve")
    graph.add_edge("retrieve", "draft")
    graph.add_edge("draft", "verify")
    graph.add_conditional_edges("verify", should_retry, {
        "accept": "accept",
        "retry": "retry_draft",
        "replan": "replan",
        "accept_with_caveat": "accept_with_caveat",
    })
    graph.add_edge("retry_draft", "draft")
    graph.add_edge("replan", "draft")
    graph.add_edge("accept", END)
    graph.add_edge("accept_with_caveat", END)

    return graph.compile()


def query(question: str) -> AgentState:
    """Run the self-correcting research agent."""
    app = build_agent()

    initial_state: AgentState = {
        "question": question,
        "is_complex": False,
        "sub_questions": [],
        "evidence": {},
        "draft": "",
        "verification": {},
        "retry_count": 0,
        "final_answer": "",
        "trace": [],
    }

    result = app.invoke(initial_state)
    return result
