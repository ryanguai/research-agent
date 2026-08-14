"""Evaluation runner — loads test set, runs pipeline, computes metrics."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import structlog

log = structlog.get_logger()


@dataclass
class EvalCase:
    question_id: str
    question: str
    category: str  # "factual", "synthesis", "adversarial"
    expected_arxiv_ids: list[str]  # papers that should be retrieved
    expected_sections: list[str]  # sections that contain the answer
    expected_answer_keywords: list[str]  # key terms the answer should contain
    should_decline: bool  # True for adversarial cases


@dataclass
class EvalResult:
    question_id: str
    category: str
    retrieved_arxiv_ids: list[str]
    retrieval_precision: float
    retrieval_recall: float
    answer: str
    citations_correct: bool
    declined_correctly: bool | None  # None for non-adversarial
    llm_judge_score: float | None
    latency_ms: float
    cost_usd: float


def load_eval_set(path: str = "eval/test_set.json") -> list[EvalCase]:
    """Load the hand-written evaluation set."""
    with open(path) as f:
        raw = json.load(f)
    return [EvalCase(**case) for case in raw]


def compute_retrieval_metrics(
    retrieved_ids: list[str], expected_ids: list[str]
) -> tuple[float, float]:
    """Compute precision and recall for retrieved documents."""
    if not expected_ids:
        return (1.0, 1.0) if not retrieved_ids else (0.0, 1.0)

    retrieved_set = set(retrieved_ids)
    expected_set = set(expected_ids)

    true_positives = len(retrieved_set & expected_set)
    precision = true_positives / len(retrieved_set) if retrieved_set else 0.0
    recall = true_positives / len(expected_set) if expected_set else 0.0

    return precision, recall


def run_eval(
    eval_set: list[EvalCase],
    pipeline_fn,
    output_path: str = "eval/results.json",
) -> list[EvalResult]:
    """Run the full eval suite and save results.

    pipeline_fn should accept a question string and return
    (answer, retrieved_chunks, generation_result).
    """
    results = []
    for case in eval_set:
        log.info("eval_running", question_id=case.question_id, category=case.category)

        start = time.time()
        answer, chunks, gen_result = pipeline_fn(case.question)
        latency = (time.time() - start) * 1000

        retrieved_ids = list({c.arxiv_id for c in chunks})
        precision, recall = compute_retrieval_metrics(retrieved_ids, case.expected_arxiv_ids)

        declined = _check_declined(answer)
        declined_correctly = None
        if case.should_decline:
            declined_correctly = declined
        elif not case.should_decline:
            declined_correctly = not declined if declined else None

        result = EvalResult(
            question_id=case.question_id,
            category=case.category,
            retrieved_arxiv_ids=retrieved_ids,
            retrieval_precision=precision,
            retrieval_recall=recall,
            answer=answer,
            citations_correct=_check_citations(answer, case.expected_arxiv_ids),
            declined_correctly=declined_correctly,
            llm_judge_score=None,  # filled in by judge pass
            latency_ms=latency,
            cost_usd=gen_result.cost_usd if gen_result else 0.0,
        )
        results.append(result)

    _save_results(results, output_path)
    _print_summary(results)
    return results


def _check_declined(answer: str) -> bool:
    decline_phrases = [
        "don't have enough information",
        "cannot answer",
        "not enough information",
        "unable to answer",
        "no relevant information",
    ]
    return any(phrase in answer.lower() for phrase in decline_phrases)


def _check_citations(answer: str, expected_ids: list[str]) -> bool:
    if not expected_ids:
        return True
    return any(aid in answer for aid in expected_ids)


def _save_results(results: list[EvalResult], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump([r.__dict__ for r in results], f, indent=2)
    log.info("eval_saved", path=path, num_results=len(results))


def _print_summary(results: list[EvalResult]) -> None:
    by_category: dict[str, list[EvalResult]] = {}
    for r in results:
        by_category.setdefault(r.category, []).append(r)

    print("\n=== EVAL SUMMARY ===")
    for cat, cat_results in by_category.items():
        avg_precision = sum(r.retrieval_precision for r in cat_results) / len(cat_results)
        avg_recall = sum(r.retrieval_recall for r in cat_results) / len(cat_results)
        avg_latency = sum(r.latency_ms for r in cat_results) / len(cat_results)
        total_cost = sum(r.cost_usd for r in cat_results)

        print(f"\n{cat.upper()} ({len(cat_results)} questions):")
        print(f"  Retrieval precision: {avg_precision:.2%}")
        print(f"  Retrieval recall:    {avg_recall:.2%}")
        print(f"  Avg latency:         {avg_latency:.0f}ms")
        print(f"  Total cost:          ${total_cost:.4f}")

        if cat == "adversarial":
            correct_declines = sum(
                1 for r in cat_results if r.declined_correctly is True
            )
            print(f"  Correct declines:    {correct_declines}/{len(cat_results)}")
    print()
