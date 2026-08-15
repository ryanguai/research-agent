"""Eval: compare self-correcting agent vs flat pipeline on complex questions."""

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

from deep_research.agent import query as agent_query
from src.eval.judge import judge_answer
from src.pipeline import Pipeline


def run_flat(question: str, pipeline: Pipeline) -> dict:
    start = time.time()
    answer, chunks, result = pipeline.query(question)
    latency = (time.time() - start) * 1000
    return {
        "answer": answer,
        "latency_ms": latency,
        "num_chunks": len(chunks),
        "papers": len(set(c.arxiv_id for c in chunks)),
    }


def run_agent(question: str) -> dict:
    start = time.time()
    result = agent_query(question)
    latency = (time.time() - start) * 1000
    return {
        "answer": result["final_answer"],
        "latency_ms": latency,
        "trace": result["trace"],
        "retry_count": result["retry_count"],
        "is_complex": result["is_complex"],
        "sub_questions": result["sub_questions"],
        "verification": result["verification"],
    }


def main():
    with open("deep_research/eval_set.json") as f:
        eval_set = json.load(f)

    pipeline = Pipeline(index_dir="data/index", retrieval_mode="hybrid", provider="gemini")

    results = []

    for i, case in enumerate(eval_set):
        if case["category"] == "verification_trap":
            continue

        log.info("eval", progress=f"{i+1}/{len(eval_set)}", question=case["question"][:60])

        time.sleep(8)
        try:
            flat_result = run_flat(case["question"], pipeline)
        except Exception as e:
            log.warning("flat_failed", error=str(e)[:80])
            flat_result = {"answer": "", "latency_ms": 0, "num_chunks": 0, "papers": 0}

        time.sleep(8)
        try:
            agent_result = run_agent(case["question"])
        except Exception as e:
            log.warning("agent_failed", error=str(e)[:80])
            agent_result = {"answer": "", "latency_ms": 0, "trace": [], "retry_count": 0}

        time.sleep(8)
        flat_judge = judge_answer(
            case["question"], flat_result["answer"],
            f"Complex research question requiring multi-paper synthesis",
            case["category"],
        )

        time.sleep(8)
        agent_judge = judge_answer(
            case["question"], agent_result["answer"],
            f"Complex research question requiring multi-paper synthesis",
            case["category"],
        )

        results.append({
            "question_id": case["question_id"],
            "question": case["question"],
            "category": case["category"],
            "flat_judge": flat_judge["score"],
            "agent_judge": agent_judge["score"],
            "flat_latency": flat_result["latency_ms"],
            "agent_latency": agent_result["latency_ms"],
            "agent_retries": agent_result.get("retry_count", 0),
            "agent_is_complex": agent_result.get("is_complex", False),
            "agent_sub_questions": agent_result.get("sub_questions", []),
            "agent_trace": agent_result.get("trace", []),
        })

    out_path = Path("deep_research/results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print_comparison(results)
    log.info("eval_complete", results_path=str(out_path))


def print_comparison(results: list[dict]):
    complex_results = [r for r in results if r["category"] == "complex"]
    simple_results = [r for r in results if r["category"] == "simple"]

    print("\n" + "=" * 70)
    print("DEEP RESEARCH AGENT vs FLAT PIPELINE")
    print("=" * 70)

    for label, subset in [("Complex questions", complex_results), ("Simple questions", simple_results)]:
        if not subset:
            continue
        flat_avg = sum(r["flat_judge"] for r in subset) / len(subset)
        agent_avg = sum(r["agent_judge"] for r in subset) / len(subset)
        flat_latency = sum(r["flat_latency"] for r in subset) / len(subset)
        agent_latency = sum(r["agent_latency"] for r in subset) / len(subset)
        retries = sum(r["agent_retries"] for r in subset)

        delta = agent_avg - flat_avg
        sign = "+" if delta > 0 else ""

        print(f"\n{label} ({len(subset)}):")
        print(f"  Judge score — Flat: {flat_avg:.2f}  Agent: {agent_avg:.2f}  Delta: {sign}{delta:.2f}")
        print(f"  Latency     — Flat: {flat_latency:.0f}ms  Agent: {agent_latency:.0f}ms")
        print(f"  Total retries: {retries}")

    print("=" * 70)


if __name__ == "__main__":
    main()
