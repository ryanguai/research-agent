"""Run the full evaluation suite: vector vs hybrid, with LLM-as-judge scoring."""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import structlog

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ]
)
log = structlog.get_logger()

from src.eval.judge import judge_answer
from src.eval.runner import EvalCase, compute_retrieval_metrics, _check_declined, _check_citations
from src.pipeline import Pipeline


def load_eval_set(path: str = "eval/test_set.json") -> list[EvalCase]:
    with open(path) as f:
        raw = json.load(f)
    return [
        EvalCase(
            question_id=q["question_id"],
            question=q["question"],
            category=q["category"],
            expected_arxiv_ids=q.get("expected_arxiv_ids", []),
            expected_sections=q.get("expected_sections", []),
            expected_answer_keywords=q.get("expected_answer_keywords", []),
            should_decline=q.get("should_decline", False),
        )
        for q in raw
    ]


def load_reference_answers(path: str = "eval/test_set.json") -> dict[str, str]:
    with open(path) as f:
        raw = json.load(f)
    return {q["question_id"]: q.get("reference_answer", "") for q in raw}


def run_single_config(
    pipeline: Pipeline,
    eval_set: list[EvalCase],
    references: dict[str, str],
    config_name: str,
) -> list[dict]:
    results = []
    for i, case in enumerate(eval_set):
        log.info(
            "eval_running",
            config=config_name,
            question_id=case.question_id,
            category=case.category,
            progress=f"{i+1}/{len(eval_set)}",
        )

        time.sleep(8)  # rate limit: stay under 15 req/min for Gemini free tier

        start = time.time()
        try:
            answer, chunks, gen_result = pipeline.query(case.question)
        except Exception as e:
            log.warning("query_failed", question_id=case.question_id, error=str(e))
            results.append({
                "question_id": case.question_id,
                "category": case.category,
                "config": config_name,
                "error": str(e),
            })
            continue
        latency = (time.time() - start) * 1000

        retrieved_ids = list({c.arxiv_id for c in chunks})
        precision, recall = compute_retrieval_metrics(retrieved_ids, case.expected_arxiv_ids)

        declined = _check_declined(answer)
        declined_correctly = None
        if case.should_decline:
            declined_correctly = declined
        elif declined:
            declined_correctly = False

        ref_answer = references.get(case.question_id, "")
        time.sleep(5)  # rate limit before judge call
        judge_result = judge_answer(
            question=case.question,
            generated_answer=answer,
            reference_answer=ref_answer,
            category=case.category,
        )

        results.append({
            "question_id": case.question_id,
            "category": case.category,
            "config": config_name,
            "answer": answer,
            "retrieved_arxiv_ids": retrieved_ids,
            "retrieval_precision": precision,
            "retrieval_recall": recall,
            "citations_correct": _check_citations(answer, case.expected_arxiv_ids),
            "declined_correctly": declined_correctly,
            "judge_score": judge_result["score"],
            "judge_reasoning": judge_result["reasoning"],
            "latency_ms": latency,
            "cost_usd": gen_result.cost_usd,
        })

    return results


def print_comparison(vector_results: list[dict], hybrid_results: list[dict]) -> str:
    """Print a comparison table and return it as a string."""

    def summarize(results: list[dict], config: str) -> dict:
        by_cat = {}
        for r in results:
            if "error" in r:
                continue
            by_cat.setdefault(r["category"], []).append(r)

        summary = {"config": config}
        for cat in ["factual", "synthesis", "adversarial"]:
            cat_results = by_cat.get(cat, [])
            if not cat_results:
                continue
            summary[f"{cat}_precision"] = sum(r["retrieval_precision"] for r in cat_results) / len(cat_results)
            summary[f"{cat}_recall"] = sum(r["retrieval_recall"] for r in cat_results) / len(cat_results)
            summary[f"{cat}_judge"] = sum(r["judge_score"] for r in cat_results) / len(cat_results)
            summary[f"{cat}_latency"] = sum(r["latency_ms"] for r in cat_results) / len(cat_results)
            summary[f"{cat}_cost"] = sum(r["cost_usd"] for r in cat_results)

            if cat == "adversarial":
                correct = sum(1 for r in cat_results if r.get("declined_correctly") is True)
                summary[f"{cat}_decline_rate"] = correct / len(cat_results)

        all_valid = [r for r in results if "error" not in r]
        summary["overall_judge"] = sum(r["judge_score"] for r in all_valid) / len(all_valid) if all_valid else 0
        summary["overall_latency"] = sum(r["latency_ms"] for r in all_valid) / len(all_valid) if all_valid else 0
        summary["total_cost"] = sum(r["cost_usd"] for r in all_valid)
        return summary

    v = summarize(vector_results, "vector")
    h = summarize(hybrid_results, "hybrid")

    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("RETRIEVAL COMPARISON: Vector-Only vs Hybrid (BM25 + Vector)")
    lines.append("=" * 80)

    header = f"{'Metric':<40} {'Vector':>12} {'Hybrid':>12} {'Delta':>12}"
    lines.append(header)
    lines.append("-" * 80)

    metrics = [
        ("Overall Judge Score (1-5)", "overall_judge"),
        ("Overall Avg Latency (ms)", "overall_latency"),
        ("Total Cost ($)", "total_cost"),
    ]

    for cat in ["factual", "synthesis", "adversarial"]:
        metrics.extend([
            (f"{cat.title()} — Retrieval Precision", f"{cat}_precision"),
            (f"{cat.title()} — Retrieval Recall", f"{cat}_recall"),
            (f"{cat.title()} — Judge Score", f"{cat}_judge"),
        ])
        if cat == "adversarial":
            metrics.append((f"{cat.title()} — Decline Rate", f"{cat}_decline_rate"))

    for label, key in metrics:
        vv = v.get(key, 0)
        hh = h.get(key, 0)
        delta = hh - vv
        sign = "+" if delta > 0 else ""

        if "latency" in key.lower():
            lines.append(f"{label:<40} {vv:>12.0f} {hh:>12.0f} {sign}{delta:>11.0f}")
        elif "cost" in key.lower():
            lines.append(f"{label:<40} {vv:>12.4f} {hh:>12.4f} {sign}{delta:>11.4f}")
        else:
            lines.append(f"{label:<40} {vv:>12.3f} {hh:>12.3f} {sign}{delta:>11.3f}")

    lines.append("=" * 80)

    report = "\n".join(lines)
    print(report)
    return report


def main():
    eval_path = "eval/test_set.json"
    if not Path(eval_path).exists():
        print(f"No eval set found at {eval_path}. Run generate_eval_candidates.py first.")
        sys.exit(1)

    eval_set = load_eval_set(eval_path)
    references = load_reference_answers(eval_path)
    log.info("eval_loaded", total=len(eval_set))

    # Run vector-only
    log.info("starting_config", config="vector")
    vector_pipeline = Pipeline(
        index_dir="data/index", retrieval_mode="vector", provider="gemini"
    )
    vector_results = run_single_config(vector_pipeline, eval_set, references, "vector")

    # Run hybrid
    log.info("starting_config", config="hybrid")
    hybrid_pipeline = Pipeline(
        index_dir="data/index", retrieval_mode="hybrid", provider="gemini"
    )
    hybrid_results = run_single_config(hybrid_pipeline, eval_set, references, "hybrid")

    # Save raw results
    out_dir = Path("eval/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "vector_results.json", "w") as f:
        json.dump(vector_results, f, indent=2)
    with open(out_dir / "hybrid_results.json", "w") as f:
        json.dump(hybrid_results, f, indent=2)

    # Print comparison
    report = print_comparison(vector_results, hybrid_results)
    with open(out_dir / "comparison_report.txt", "w") as f:
        f.write(report)

    log.info("eval_complete", results_dir=str(out_dir))


if __name__ == "__main__":
    main()
