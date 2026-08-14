"""Basic concurrency/load test for the pipeline."""

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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

from src.pipeline import Pipeline

QUERIES = [
    "What is retrieval-augmented generation?",
    "How does BM25 compare to dense retrieval?",
    "What are knowledge poisoning attacks in RAG?",
    "How can caching improve RAG latency?",
    "What chunking strategies are used in RAG systems?",
    "How do rerankers improve retrieval quality?",
    "What is adaptive retrieval?",
    "How do RAG systems handle multi-hop questions?",
    "What evaluation metrics are used for RAG?",
    "How does hybrid retrieval work?",
    "What are the challenges of deploying RAG in production?",
    "How do RAG systems cite their sources?",
]


def run_query(pipeline: Pipeline, query: str, query_id: int) -> dict:
    start = time.time()
    try:
        answer, chunks, result = pipeline.query(query)
        latency = (time.time() - start) * 1000
        return {
            "query_id": query_id,
            "query": query,
            "status": "success",
            "latency_ms": latency,
            "chunks": len(chunks),
            "answer_len": len(answer),
        }
    except Exception as e:
        latency = (time.time() - start) * 1000
        return {
            "query_id": query_id,
            "query": query,
            "status": "error",
            "error": str(e),
            "latency_ms": latency,
        }


def main():
    concurrency_levels = [1, 4, 8]
    pipeline = Pipeline(index_dir="data/index", retrieval_mode="vector", provider="ollama")

    all_results = {}

    for n_workers in concurrency_levels:
        print(f"\n{'='*60}")
        print(f"Testing with {n_workers} concurrent requests")
        print(f"{'='*60}")

        results = []
        start = time.time()

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(run_query, pipeline, q, i): i
                for i, q in enumerate(QUERIES)
            }

            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                status = "OK" if result["status"] == "success" else "FAIL"
                print(f"  [{status}] Query {result['query_id']}: {result['latency_ms']:.0f}ms")

        total_time = (time.time() - start) * 1000
        successes = [r for r in results if r["status"] == "success"]
        failures = [r for r in results if r["status"] == "error"]

        latencies = [r["latency_ms"] for r in successes]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        p95 = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0

        summary = {
            "concurrency": n_workers,
            "total_queries": len(QUERIES),
            "successes": len(successes),
            "failures": len(failures),
            "total_time_ms": total_time,
            "avg_latency_ms": avg_latency,
            "p95_latency_ms": p95,
            "throughput_qps": len(QUERIES) / (total_time / 1000) if total_time > 0 else 0,
        }
        all_results[n_workers] = summary

        print(f"\n  Successes: {len(successes)}/{len(QUERIES)}")
        print(f"  Avg latency: {avg_latency:.0f}ms")
        print(f"  P95 latency: {p95:.0f}ms")
        print(f"  Total time:  {total_time:.0f}ms")
        print(f"  Throughput:  {summary['throughput_qps']:.2f} queries/sec")

    # Save results
    out_path = Path("eval/results/load_test.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
