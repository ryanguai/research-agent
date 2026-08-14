"""Generate candidate Q&A pairs from the indexed corpus for the eval set.

Samples papers, reads their chunks, and uses the LLM to propose questions
in three categories: factual, synthesis, and adversarial.

The output is a draft — the project owner should review and edit before
using as ground truth.
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb
import structlog
from openai import OpenAI

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ]
)
log = structlog.get_logger()


def get_paper_summaries(collection: chromadb.Collection) -> dict[str, dict]:
    """Group chunks by paper and extract metadata."""
    all_data = collection.get(include=["metadatas", "documents"])
    papers = {}
    for i, meta in enumerate(all_data["metadatas"]):
        aid = meta["arxiv_id"]
        if aid not in papers:
            papers[aid] = {
                "arxiv_id": aid,
                "title": meta["paper_title"],
                "sections": {},
                "chunk_count": 0,
            }
        sec = meta["section_name"]
        if sec not in papers[aid]["sections"]:
            papers[aid]["sections"][sec] = []
        papers[aid]["sections"][sec].append(all_data["documents"][i][:500])
        papers[aid]["chunk_count"] += 1
    return papers


def generate_factual_questions(client: OpenAI, paper: dict, n: int = 3) -> list[dict]:
    """Generate factual questions about a single paper."""
    sections_text = ""
    for sec_name, chunks in list(paper["sections"].items())[:4]:
        sections_text += f"\n--- {sec_name} ---\n"
        sections_text += chunks[0] if chunks else ""

    resp = client.chat.completions.create(
        model="llama3.2",
        messages=[
            {
                "role": "system",
                "content": (
                    "Generate factual questions that can be answered from the paper excerpts below. "
                    "Each question should have a clear, specific answer found in the text. "
                    "Return ONLY a JSON array of objects with keys: "
                    '"question", "answer", "section" (which section contains the answer). '
                    "No markdown, no explanation, just the JSON array."
                ),
            },
            {
                "role": "user",
                "content": f"Paper: {paper['title']} (arXiv: {paper['arxiv_id']})\n{sections_text}",
            },
        ],
        temperature=0.3,
    )

    try:
        text = resp.choices[0].message.content
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            questions = json.loads(text[start:end])
            return [
                {
                    "question": q["question"],
                    "category": "factual",
                    "expected_arxiv_ids": [paper["arxiv_id"]],
                    "expected_sections": [q.get("section", "")],
                    "expected_answer_keywords": q.get("answer", "").split()[:10],
                    "should_decline": False,
                    "reference_answer": q.get("answer", ""),
                }
                for q in questions[:n]
            ]
    except (json.JSONDecodeError, KeyError) as e:
        log.warning("parse_failed", paper=paper["arxiv_id"], error=str(e))
    return []


def generate_synthesis_questions(
    client: OpenAI, paper_a: dict, paper_b: dict, n: int = 2
) -> list[dict]:
    """Generate questions requiring info from two papers."""
    text_a = list(paper_a["sections"].values())[0][0][:400] if paper_a["sections"] else ""
    text_b = list(paper_b["sections"].values())[0][0][:400] if paper_b["sections"] else ""

    resp = client.chat.completions.create(
        model="llama3.2",
        messages=[
            {
                "role": "system",
                "content": (
                    "Generate questions that require combining information from BOTH papers below. "
                    "Good synthesis questions ask about comparisons, contrasts, or complementary ideas. "
                    "Return ONLY a JSON array of objects with keys: "
                    '"question", "answer". No markdown, no explanation, just the JSON array.'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Paper A: {paper_a['title']} (arXiv: {paper_a['arxiv_id']})\n{text_a}\n\n"
                    f"Paper B: {paper_b['title']} (arXiv: {paper_b['arxiv_id']})\n{text_b}"
                ),
            },
        ],
        temperature=0.3,
    )

    try:
        text = resp.choices[0].message.content
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            questions = json.loads(text[start:end])
            return [
                {
                    "question": q["question"],
                    "category": "synthesis",
                    "expected_arxiv_ids": [paper_a["arxiv_id"], paper_b["arxiv_id"]],
                    "expected_sections": [],
                    "expected_answer_keywords": q.get("answer", "").split()[:10],
                    "should_decline": False,
                    "reference_answer": q.get("answer", ""),
                }
                for q in questions[:n]
            ]
    except (json.JSONDecodeError, KeyError) as e:
        log.warning("parse_failed", error=str(e))
    return []


ADVERSARIAL_QUESTIONS = [
    {
        "question": "What is the current stock price of NVIDIA?",
        "category": "adversarial",
        "expected_arxiv_ids": [],
        "expected_sections": [],
        "expected_answer_keywords": [],
        "should_decline": True,
        "reference_answer": "Out of scope — not covered by the corpus.",
    },
    {
        "question": "How do I fine-tune GPT-4 on my custom dataset?",
        "category": "adversarial",
        "expected_arxiv_ids": [],
        "expected_sections": [],
        "expected_answer_keywords": [],
        "should_decline": True,
        "reference_answer": "Out of scope — fine-tuning is not a topic in this RAG-focused corpus.",
    },
    {
        "question": "What did Yann LeCun say about RAG at NeurIPS 2024?",
        "category": "adversarial",
        "expected_arxiv_ids": [],
        "expected_sections": [],
        "expected_answer_keywords": [],
        "should_decline": True,
        "reference_answer": "Out of scope — specific conference talks are not in the corpus.",
    },
    {
        "question": "Compare the performance of Claude and GPT-4 on medical question answering.",
        "category": "adversarial",
        "expected_arxiv_ids": [],
        "expected_sections": [],
        "expected_answer_keywords": [],
        "should_decline": True,
        "reference_answer": "Out of scope — medical QA is not covered by this RAG-focused corpus.",
    },
    {
        "question": "What is the best programming language for building RAG systems?",
        "category": "adversarial",
        "expected_arxiv_ids": [],
        "expected_sections": [],
        "expected_answer_keywords": [],
        "should_decline": True,
        "reference_answer": "Out of scope — opinion question not answerable from research papers.",
    },
    {
        "question": "How much does it cost to run a RAG system in production at Google scale?",
        "category": "adversarial",
        "expected_arxiv_ids": [],
        "expected_sections": [],
        "expected_answer_keywords": [],
        "should_decline": True,
        "reference_answer": "Out of scope — proprietary cost data is not in arXiv papers.",
    },
    {
        "question": "What will be the next breakthrough in RAG research in 2027?",
        "category": "adversarial",
        "expected_arxiv_ids": [],
        "expected_sections": [],
        "expected_answer_keywords": [],
        "should_decline": True,
        "reference_answer": "Out of scope — future predictions cannot be answered from existing papers.",
    },
    {
        "question": "Explain how blockchain technology can improve RAG systems.",
        "category": "adversarial",
        "expected_arxiv_ids": [],
        "expected_sections": [],
        "expected_answer_keywords": [],
        "should_decline": True,
        "reference_answer": "Out of scope — blockchain is not a topic in this corpus.",
    },
    {
        "question": "What are the ethical implications of using RAG in healthcare?",
        "category": "adversarial",
        "expected_arxiv_ids": [],
        "expected_sections": [],
        "expected_answer_keywords": [],
        "should_decline": True,
        "reference_answer": "Out of scope — healthcare ethics is not covered in this technical corpus.",
    },
    {
        "question": "How does Apple's Siri use retrieval-augmented generation internally?",
        "category": "adversarial",
        "expected_arxiv_ids": [],
        "expected_sections": [],
        "expected_answer_keywords": [],
        "should_decline": True,
        "reference_answer": "Out of scope — proprietary system details are not in arXiv papers.",
    },
]


def main():
    client_ollama = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

    db = chromadb.PersistentClient(path="data/index")
    collection = db.get_collection("papers")

    log.info("loading_papers")
    papers = get_paper_summaries(collection)
    paper_list = list(papers.values())
    log.info("papers_loaded", count=len(paper_list))

    all_questions = []

    # Factual questions: sample 20 papers, 3 questions each = ~60
    sampled = random.sample(paper_list, min(20, len(paper_list)))
    for i, paper in enumerate(sampled):
        log.info("generating_factual", paper=paper["arxiv_id"], progress=f"{i+1}/{len(sampled)}")
        qs = generate_factual_questions(client_ollama, paper, n=3)
        all_questions.extend(qs)
        log.info("generated", count=len(qs), total=len(all_questions))

    # Synthesis questions: sample 10 pairs, 2 questions each = ~20
    for i in range(min(10, len(paper_list) // 2)):
        pair = random.sample(paper_list, 2)
        log.info("generating_synthesis", progress=f"{i+1}/10")
        qs = generate_synthesis_questions(client_ollama, pair[0], pair[1], n=2)
        all_questions.extend(qs)

    # Adversarial questions: hand-written
    all_questions.extend(ADVERSARIAL_QUESTIONS)

    # Assign IDs
    for i, q in enumerate(all_questions):
        q["question_id"] = f"{q['category']}_{i:03d}"

    # Save
    out_path = Path("eval/test_set.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_questions, f, indent=2)

    by_cat = {}
    for q in all_questions:
        by_cat.setdefault(q["category"], []).append(q)

    print(f"\n=== Generated {len(all_questions)} questions ===")
    for cat, qs in by_cat.items():
        print(f"  {cat}: {len(qs)}")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
