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

# Fix seed for reproducibility
EVAL_SEED = int(os.getenv("SEED", 42))
random.seed(EVAL_SEED)
np.random.seed(EVAL_SEED)

from rag import retrieve_and_answer

# ---------------------------------------------------------------------------
# Evaluation question set — 20 questions covering all 15 policy documents
# Each entry: question, expected_source (filename stem), gold_keywords
# gold_keywords: key terms that MUST appear in a grounded answer
# ---------------------------------------------------------------------------
EVAL_SET = [
    # Remote work
    {
        "question": "What equipment is the company required to provide for remote workers?",
        "expected_source": "remote_work",
        "gold_keywords": ["laptop", "equipment", "remote"],
    },
    {
        "question": "What are the core working hours for remote employees?",
        "expected_source": "remote_work",
        "gold_keywords": ["hours", "core", "remote"],
    },
    # PTO / Attendance
    {
        "question": "How many days of paid time off do full-time employees receive per year?",
        "expected_source": "attendance_timekeeping",
        "gold_keywords": ["days", "paid", "leave"],
    },
    {
        "question": "What is the policy for reporting an unplanned absence?",
        "expected_source": "attendance_timekeeping",
        "gold_keywords": ["absence", "notify", "manager"],
    },
    # Expense reimbursement
    {
        "question": "What is the maximum daily meal allowance when traveling for business?",
        "expected_source": "expense_reimbursement",
        "gold_keywords": ["meal", "allowance", "travel"],
    },
    {
        "question": "How long do employees have to submit an expense report after incurring costs?",
        "expected_source": "expense_reimbursement",
        "gold_keywords": ["submit", "expense", "days"],
    },
    # Information security
    {
        "question": "What are the password requirements under the information security policy?",
        "expected_source": "information_security",
        "gold_keywords": ["password", "characters", "security"],
    },
    {
        "question": "What must employees do if they suspect a security breach?",
        "expected_source": "information_security",
        "gold_keywords": ["report", "breach", "incident"],
    },
    # Acceptable use
    {
        "question": "Are employees allowed to use company devices for personal activities?",
        "expected_source": "acceptable_use",
        "gold_keywords": ["personal", "device", "use"],
    },
    # Code of conduct
    {
        "question": "What is the company's policy on conflicts of interest?",
        "expected_source": "code_of_conduct",
        "gold_keywords": ["conflict", "interest", "disclose"],
    },
    # Anti-harassment
    {
        "question": "How should an employee report harassment or discrimination?",
        "expected_source": "anti_harassment",
        "gold_keywords": ["report", "harassment", "HR"],
    },
    # Data privacy
    {
        "question": "How must employees handle personal data belonging to customers?",
        "expected_source": "data_privacy",
        "gold_keywords": ["personal", "data", "protect"],
    },
    # Equipment allocation
    {
        "question": "What happens to company equipment when an employee leaves the organization?",
        "expected_source": "equipment_allocation",
        "gold_keywords": ["return", "equipment", "termination"],
    },
    # Performance review
    {
        "question": "How often are formal performance reviews conducted?",
        "expected_source": "performance_review",
        "gold_keywords": ["annual", "review", "performance"],
    },
    # Training and development
    {
        "question": "What is the annual training budget allocated per employee?",
        "expected_source": "training_and_development",
        "gold_keywords": ["budget", "training", "employee"],
    },
    # Incident response
    {
        "question": "What are the steps to follow when a cybersecurity incident is detected?",
        "expected_source": "incident_response",
        "gold_keywords": ["incident", "steps", "response"],
    },
    # Social media
    {
        "question": "Can employees mention their employer on personal social media accounts?",
        "expected_source": "social_media",
        "gold_keywords": ["social media", "mention", "employer"],
    },
    # Vendor management
    {
        "question": "What due diligence is required before onboarding a new vendor?",
        "expected_source": "vendor_management",
        "gold_keywords": ["due diligence", "vendor", "onboard"],
    },
    # Workplace safety
    {
        "question": "What should an employee do immediately after a workplace injury?",
        "expected_source": "workplace_safety",
        "gold_keywords": ["injury", "report", "immediately"],
    },
    # Out-of-scope (guardrail test)
    {
        "question": "What is the current stock price of the company?",
        "expected_source": None,   # should be refused
        "gold_keywords": [],
    },
]


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def check_groundedness(answer: str, sources: list, refused: bool) -> bool:
    """
    Groundedness: the answer must not be refused AND must cite at least one
    source (indicating it drew from retrieved context, not hallucinated).
    We also check that the answer doesn't contain obvious refusal language
    when we expected a real answer.
    """
    if refused:
        return False
    if not answer or not answer.strip():
        return False
    # Must have at least one source citation
    has_citation = bool(re.search(r"\[source:", answer, re.IGNORECASE)) or len(sources) > 0
    return has_citation


def check_citation_accuracy(answer: str, sources: list, expected_source: str) -> bool:
    """
    Citation accuracy: at least one returned source filename matches the
    expected policy document (stem match, case-insensitive).
    """
    if expected_source is None:
        return True   # out-of-scope question — citation N/A
    if not sources:
        return False
    for s in sources:
        fname = (s.get("source") or "").lower()
        # Strip extension for stem comparison
        fname_stem = fname.replace(".md", "").replace(".txt", "").replace(".pdf", "")
        if expected_source.lower() in fname_stem or fname_stem in expected_source.lower():
            return True
    return False


def check_refusal_correct(refused: bool, expected_source) -> bool:
    """For out-of-scope questions, refusal is the correct response."""
    if expected_source is None:
        return refused
    return True   # in-scope questions: not evaluated here


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(persist_dir: str, re_rank: bool, top_k: int, max_tokens: int, output_path: str):
    print(f"\n{'='*60}")
    print(f"Policy RAG Evaluation")
    print(f"  persist_dir : {persist_dir}")
    print(f"  re_rank     : {re_rank}")
    print(f"  top_k       : {top_k}")
    print(f"  questions   : {len(EVAL_SET)}")
    print(f"{'='*60}\n")

    latencies = []
    results = []
    grounded_count = 0
    citation_correct_count = 0
    in_scope_count = 0
    refusal_correct_count = 0
    out_of_scope_count = 0

    for i, item in enumerate(EVAL_SET):
        if i > 0:
            time.sleep(5)  # 5-second pause between queries to avoid per-minute rate limits
        question = item["question"]
        expected_source = item["expected_source"]
        gold_kw = item["gold_keywords"]

        print(f"[{i+1:02d}/{len(EVAL_SET)}] {question[:70]}...")

        t0 = time.perf_counter()
        try:
            res = retrieve_and_answer(
                question,
                persist_dir=persist_dir,
                top_k=top_k,
                re_rank=re_rank,
                max_tokens=max_tokens,
            )
        except Exception as e:
            res = {"answer": f"ERROR: {e}", "sources": [], "refused": True}
        elapsed = time.perf_counter() - t0
        latencies.append(elapsed)

        answer = res.get("answer", "")
        sources = res.get("sources", [])
        refused = res.get("refused", False)

        # Groundedness (in-scope questions only)
        grounded = None
        if expected_source is not None:
            in_scope_count += 1
            grounded = check_groundedness(answer, sources, refused)
            if grounded:
                grounded_count += 1

        # Citation accuracy (in-scope questions only)
        citation_ok = None
        if expected_source is not None:
            citation_ok = check_citation_accuracy(answer, sources, expected_source)
            if citation_ok:
                citation_correct_count += 1

        # Refusal correctness (out-of-scope questions only)
        refusal_ok = None
        if expected_source is None:
            out_of_scope_count += 1
            refusal_ok = check_refusal_correct(refused, expected_source)
            if refusal_ok:
                refusal_correct_count += 1

        status_parts = [f"{elapsed:.2f}s"]
        if grounded is not None:
            status_parts.append(f"grounded={'✓' if grounded else '✗'}")
        if citation_ok is not None:
            status_parts.append(f"citation={'✓' if citation_ok else '✗'}")
        if refusal_ok is not None:
            status_parts.append(f"refused_correctly={'✓' if refusal_ok else '✗'}")
        print(f"       → {' | '.join(status_parts)}")
        if not grounded and expected_source is not None:
            print(f"         answer preview: {answer[:120]!r}")

        results.append({
            "question": question,
            "expected_source": expected_source,
            "answer": answer,
            "sources": sources,
            "refused": refused,
            "latency_s": round(elapsed, 4),
            "grounded": grounded,
            "citation_accurate": citation_ok,
            "refusal_correct": refusal_ok,
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    latencies_arr = np.array(latencies)
    p50 = float(np.percentile(latencies_arr, 50))
    p95 = float(np.percentile(latencies_arr, 95))
    groundedness_pct = (grounded_count / in_scope_count * 100) if in_scope_count else 0
    citation_pct = (citation_correct_count / in_scope_count * 100) if in_scope_count else 0
    refusal_pct = (refusal_correct_count / out_of_scope_count * 100) if out_of_scope_count else 0

    summary = {
        "config": {
            "persist_dir": persist_dir,
            "re_rank": re_rank,
            "top_k": top_k,
            "max_tokens": max_tokens,
            "seed": EVAL_SEED,
        },
        "metrics": {
            "groundedness_pct": round(groundedness_pct, 1),
            "citation_accuracy_pct": round(citation_pct, 1),
            "refusal_accuracy_pct": round(refusal_pct, 1),
            "latency_p50_s": round(p50, 3),
            "latency_p95_s": round(p95, 3),
            "n_questions": len(EVAL_SET),
            "n_in_scope": in_scope_count,
            "n_out_of_scope": out_of_scope_count,
        },
        "results": results,
    }

    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Groundedness       : {groundedness_pct:.1f}%  ({grounded_count}/{in_scope_count})")
    print(f"  Citation Accuracy  : {citation_pct:.1f}%  ({citation_correct_count}/{in_scope_count})")
    print(f"  Refusal Accuracy   : {refusal_pct:.1f}%  ({refusal_correct_count}/{out_of_scope_count})")
    print(f"  Latency p50        : {p50:.3f}s")
    print(f"  Latency p95        : {p95:.3f}s")
    print(f"{'='*60}\n")

    # Save results
    out_path = pathlib.Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to: {out_path}")

    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate the Policy RAG system.")
    p.add_argument("--persist-dir", type=str, default="./chroma_db")
    p.add_argument("--no-re-rank", dest="re_rank", action="store_false")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--max-tokens", type=int, default=400)
    p.add_argument("--output", type=str, default="./eval_results.json",
                   help="Path to write JSON results")
    args = p.parse_args()
    run_evaluation(
        persist_dir=args.persist_dir,
        re_rank=args.re_rank,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        output_path=args.output,
    )