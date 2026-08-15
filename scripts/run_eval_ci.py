"""CI eval: run adversarial subset only (10 questions, ~2 min, ~10 API calls)."""

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

from src.eval.runner import EvalCase, compute_retrieval_metrics, _check_declined
from src.pipeline import Pipeline


def main():
    eval_path = "eval/test_set.json"
    if not Path(eval_path).exists():
        print(f"No eval set found at {eval_path}")
        sys.exit(1)

    with open(eval_path) as f:
        raw = json.load(f)

    adversarial = [
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
        if q["category"] == "adversarial"
    ]

    log.info("ci_eval_start", num_questions=len(adversarial))

    pipeline = Pipeline(
        index_dir="data/index", retrieval_mode="vector", provider="gemini"
    )

    results = []
    for i, case in enumerate(adversarial):
        time.sleep(8)
        log.info("ci_eval_running", progress=f"{i+1}/{len(adversarial)}", question=case.question[:60])

        try:
            answer, chunks, gen_result = pipeline.query(case.question)
            declined = _check_declined(answer)
            results.append({
                "question_id": case.question_id,
                "question": case.question,
                "declined": declined,
                "answer_preview": answer[:200],
            })
        except Exception as e:
            log.warning("ci_eval_failed", question_id=case.question_id, error=str(e)[:80])
            results.append({
                "question_id": case.question_id,
                "question": case.question,
                "declined": False,
                "error": str(e)[:200],
            })

    out_path = Path("eval/results/ci_eval.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    decline_count = sum(1 for r in results if r.get("declined"))
    total = len(results)
    decline_rate = decline_count / total if total > 0 else 0

    print(f"\n=== CI EVAL RESULTS ===")
    print(f"Adversarial decline rate: {decline_count}/{total} ({decline_rate:.0%})")
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
