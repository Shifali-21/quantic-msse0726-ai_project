"""
Evaluation script for the Policy RAG system.

Metrics:
  - Groundedness: % of answers whose content is supported by the retrieved context
  - Citation Accuracy: % of answers with citations that match actual source files
  - Latency: p50 and p95 response times (seconds)

Usage (from project root, with venv activated):
    PYTHONPATH=. python src/evaluate.py --persist-dir ./chroma_db
    PYTHONPATH=. python src/evaluate.py --persist-dir ./chroma_db --no-re-rank
    PYTHONPATH=. python src/evaluate.py --persist-dir ./chroma_db --output eval_results.json
"""

import os
import sys
import json
import time
import random
import argparse
import pathlib
import re
import numpy as np
from dotenv import load_dotenv
load_dotenv()

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = str(ROOT / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from rag import retrieve_and_answer

# Fixed evaluation set - all questions answerable from corpus
EVAL_SET = [
    # 1. remote_work.md
    {
        "question": "What type of device must remote workers use?",
        "expected_source": "remote_work",
    },
    # 2. remote_work.md
    {
        "question": "What are the communication expectations for remote employees?",
        "expected_source": "remote_work",
    },
    # 3. attendance_timekeeping.md
    {
        "question": "What are the standard working hours?",
        "expected_source": "attendance_timekeeping",
    },
    # 4. attendance_timekeeping.md
    {
        "question": "How much advance notice is required before an absence?",
        "expected_source": "attendance_timekeeping",
    },
    # 5. expense_reimbursement.md
    {
        "question": "What types of expenses are allowable for reimbursement?",
        "expected_source": "expense_reimbursement",
    },
    # 6. expense_reimbursement.md
    {
        "question": "Within how many days must reimbursement claims be submitted?",
        "expected_source": "expense_reimbursement",
    },
    # 7. information_security.md
    {
        "question": "What authentication method is required for system access?",
        "expected_source": "information_security",
    },
    # 8. information_security.md
    {
        "question": "Who should employees report security incidents to?",
        "expected_source": "information_security",
    },
    # 9. acceptable_use.md
    {
        "question": "What types of websites are employees prohibited from accessing?",
        "expected_source": "acceptable_use",
    },
    # 10. acceptable_use.md
    {
        "question": "Where should employees store their work files?",
        "expected_source": "acceptable_use",
    },
    # 11. anti_harassment.md
    {
        "question": "What are three ways employees can report harassment concerns?",
        "expected_source": "anti_harassment",
    },
    # 12. data_privacy.md
    {
        "question": "What principle should guide how much customer data employees access?",
        "expected_source": "data_privacy",
    },
    # 13. equipment_allocation.md
    {
        "question": "What must employees do when equipment is damaged or lost?",
        "expected_source": "equipment_allocation",
    },
    # 14. performance_review.md
    {
        "question": "What are the main components of the performance review process?",
        "expected_source": "performance_review",
    },
    # 15. incident_response.md
    {
        "question": "What are the four steps in the incident response process?",
        "expected_source": "incident_response",
    },
    # 16. social_media.md
    {
        "question": "Can employees use personal social media accounts to represent the company?",
        "expected_source": "social_media",
    },
    # 17. training_and_development.md
    {
        "question": "What types of core training are required for all employees?",
        "expected_source": "training_and_development",
    },
    # 18. vendor_management.md
    {
        "question": "What must be done with organizational data when ending a vendor relationship?",
        "expected_source": "vendor_management",
    },
    # 19. workplace_safety.md
    {
        "question": "What should employees do if they identify a workplace hazard?",
        "expected_source": "workplace_safety",
    },
    # 20. code_of_conduct.md (guardrail test - out of scope)
    {
        "question": "What is the current stock price of the company?",
        "expected_source": None,  # Out of scope - should refuse
    },
]


def check_grounded(answer: str) -> bool:
    """Check if answer contains citation (indicates grounded in retrieved docs)."""
    return "[source:" in answer.lower()


def check_citation_accurate(answer: str, expected_source: str) -> bool:
    """Check if answer cites the expected source document."""
    if not expected_source:
        return False
    # Look for the expected filename in citations
    pattern = rf"\[source:\s*{re.escape(expected_source)}[._]"
    return bool(re.search(pattern, answer, re.IGNORECASE))


def check_refusal_correct(answer: str, refused: bool, expected_source: str) -> bool | None:
    """Check if out-of-scope questions are correctly refused."""
    if expected_source is None:  # Out-of-scope question
        return refused or ("can only answer" in answer.lower() and "policies" in answer.lower())
    return None  # Not an out-of-scope question


def run_evaluation(persist_dir: str, re_rank: bool, top_k: int):
    """Run evaluation on the fixed question set."""
    random.seed(42)
    np.random.seed(42)

    results = []
    latencies = []

    print("\n" + "=" * 60)
    print("Policy RAG Evaluation (FIXED)")
    print(f"  persist_dir : {persist_dir}")
    print(f"  re_rank     : {re_rank}")
    print(f"  top_k       : {top_k}")
    print(f"  questions   : {len(EVAL_SET)}")
    print("=" * 60 + "\n")

    for i, item in enumerate(EVAL_SET):
        # Sleep between queries to avoid rate limits (skip first question)
        if i > 0:
            time.sleep(5)

        question = item["question"]
        expected_source = item["expected_source"]
        q_display = question if len(question) <= 70 else question[:70] + "..."
        print(f"[{i+1:02d}/{len(EVAL_SET)}] {q_display}", end="", flush=True)

        start = time.time()
        try:
            res = retrieve_and_answer(
                question,
                persist_dir=persist_dir,
                top_k=top_k,
                re_rank=re_rank,
            )
            answer = res.get("answer", "")
            sources = res.get("sources", [])
            refused = res.get("refused", False)
        except Exception as e:
            print(f"\n  ERROR: {e}")
            answer = ""
            sources = []
            refused = True

        latency = time.time() - start
        latencies.append(latency)

        # Evaluate answer quality
        grounded = check_grounded(answer)
        citation_accurate = check_citation_accurate(answer, expected_source) if expected_source else False
        refusal_correct = check_refusal_correct(answer, refused, expected_source)

        # Store result
        results.append({
            "question": question,
            "expected_source": expected_source,
            "answer": answer,
            "sources": sources,
            "refused": refused,
            "latency_s": round(latency, 2),
            "grounded": grounded,
            "citation_accurate": citation_accurate,
            "refusal_correct": refusal_correct,
        })

        # Print inline summary
        status_parts = []
        status_parts.append(f"{latency:.2f}s")
        if expected_source:
            status_parts.append(f"grounded={'✓' if grounded else '✗'}")
            status_parts.append(f"citation={'✓' if citation_accurate else '✗'}")
        else:
            status_parts.append(f"refused_correctly={'✓' if refusal_correct else '✗'}")

        print(f"\n       → {' | '.join(status_parts)}")

        # Preview answer if not grounded/cited correctly
        if expected_source and (not grounded or not citation_accurate):
            preview = answer[:100].replace("\n", " ")
            print(f"         answer preview: \"{preview}...\"")

    # Compute summary stats
    in_scope = [r for r in results if r["expected_source"] is not None]
    out_of_scope = [r for r in results if r["expected_source"] is None]

    grounded_count = sum(1 for r in in_scope if r["grounded"])
    citation_count = sum(1 for r in in_scope if r["citation_accurate"])
    refusal_count = sum(1 for r in out_of_scope if r["refusal_correct"])

    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)

    summary = {
        "total_questions": len(EVAL_SET),
        "in_scope_questions": len(in_scope),
        "out_of_scope_questions": len(out_of_scope),
        "groundedness_pct": round(100 * grounded_count / len(in_scope), 1) if in_scope else 0,
        "citation_accuracy_pct": round(100 * citation_count / len(in_scope), 1) if in_scope else 0,
        "refusal_accuracy_pct": round(100 * refusal_count / len(out_of_scope), 1) if out_of_scope else 0,
        "latency_p50_s": round(p50, 3),
        "latency_p95_s": round(p95, 3),
        "results": results,
    }

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Groundedness       : {summary['groundedness_pct']}%  ({grounded_count}/{len(in_scope)})")
    print(f"  Citation Accuracy  : {summary['citation_accuracy_pct']}%  ({citation_count}/{len(in_scope)})")
    print(f"  Refusal Accuracy   : {summary['refusal_accuracy_pct']}%  ({refusal_count}/{len(out_of_scope)})")
    print(f"  Latency p50        : {summary['latency_p50_s']}s")
    print(f"  Latency p95        : {summary['latency_p95_s']}s")
    print("=" * 60 + "\n")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--persist-dir", type=str, default="./chroma_db")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-re-rank", dest="re_rank", action="store_false", default=True)
    parser.add_argument("--output", type=str, default="eval_results.json")
    args = parser.parse_args()

    summary = run_evaluation(
        persist_dir=args.persist_dir,
        re_rank=args.re_rank,
        top_k=args.top_k,
    )

    # Save results
    out_path = pathlib.Path(args.output)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Results saved to: {out_path}\n")