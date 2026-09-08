"""Execute real local embedding retrieval contracts; no simulated model fallback.

Usage: python scripts/validate_local_embeddings.py --output /tmp/vector-report.json
Install requirements-advanced.txt first. Nonzero exit means unavailable or failed.
The fixed synthetic cases are acceptance challenges, not a training/holdout set.
Nearest-neighbor retrieval does not establish the truth of an employee assessment.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version, PackageNotFoundError
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RECORDS = [
    {"id": "python", "text": "Alex built Python data pipelines, automated payroll reconciliations and repaired SQL reporting queries."},
    {"id": "sales", "text": "Blair exceeded enterprise sales targets and renewed customer contracts through careful account management."},
    {"id": "onboarding", "text": "Casey welcomed new employees, coordinated onboarding checklists and resolved questions about benefits enrollment."},
    {"id": "mentoring", "text": "Drew coached junior engineers every week and helped new developers improve through patient technical mentoring."},
    {"id": "no_mentoring", "text": "Ellis did not coach junior engineers and provided no technical mentoring to new developers."},
    {"id": "spanish_payroll", "text": "Francis corrigió errores de nómina y verificó las cotizaciones antes del cierre mensual."},
    {"id": "spanish_onboarding", "text": "Gabriel organizó la incorporación de nuevas personas en Varsovia y explicó sus beneficios durante la bienvenida."},
    {"id": "facilities", "text": "Harper maintained office equipment, repaired meeting room furniture and coordinated building access."},
]
CASES = [
    {"id": "english_technical", "query": "Who developed Python automation for payroll data?", "expected": "python", "scope": "english"},
    {"id": "english_sales", "query": "Who succeeded in customer renewals and enterprise selling?", "expected": "sales", "scope": "english"},
    {"id": "english_benefits", "query": "Who helped new starters enroll in employee benefits?", "expected": "onboarding", "scope": "english"},
    {"id": "english_facilities", "query": "Who repairs office furniture and manages building access?", "expected": "facilities", "scope": "english"},
    {"id": "positive_mentoring", "query": "Who actively coached junior engineers?", "expected": "mentoring", "scope": "negation"},
    {"id": "negative_mentoring", "query": "Who did not provide mentoring to junior developers?", "expected": "no_mentoring", "scope": "negation"},
    {"id": "spanish_payroll", "query": "¿Quién corrigió errores de nómina?", "expected": "spanish_payroll", "scope": "spanish"},
    {"id": "cross_language_payroll", "query": "Who fixed payroll mistakes and verified social contributions before month-end?", "expected": "spanish_payroll", "scope": "cross_language"},
    {"id": "cross_language_onboarding", "query": "Who organized employee orientation and explained benefits in Warsaw?", "expected": "spanish_onboarding", "scope": "cross_language"},
]
SCORE_SEMANTICS = "inverse_squared_l2_distance_not_probability_or_validated_relevance"


def judge(case: dict, results: list[dict]) -> dict:
    """Compare retrieved IDs with independently specified answers, without an LLM judge."""
    ids = [row.get("id") for row in results]
    return {**case, "retrieved_ids": ids, "passed": bool(ids) and ids[0] == case["expected"],
            "scores": [row.get("similarity_score") for row in results]}


def evaluate(model_name: str) -> dict:
    report = {
        "schema_version": 1, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": model_name, "execution": "real_sentence_transformers_and_faiss",
        "status": "blocked", "checks": [],
        "engine_sha256": hashlib.sha256((ROOT / "src/vector_engine.py").read_bytes()).hexdigest(),
        "corpus_sha256": hashlib.sha256(json.dumps({"records": RECORDS, "cases": CASES}, sort_keys=True).encode()).hexdigest(),
        "limitations": ["Small synthetic acceptance corpus; not population-level validity or an untouched final holdout.",
                        "Nearest neighbors are candidate source texts, not verified assessments or probabilities.",
                        "No validated no-match threshold; off-topic queries may retrieve unrelated records.",
                        "Passing on this model does not validate a different model or revision."],
    }
    try:
        report["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        report["commit"] = "unavailable"
    report["dependencies"] = {}
    for package in ("faiss-cpu", "sentence-transformers", "torch", "transformers",
                    "numpy", "scikit-learn", "scipy", "PyYAML"):
        try:
            report["dependencies"][package] = version(package)
        except PackageNotFoundError:
            report["dependencies"][package] = "missing"
    try:
        from src.vector_engine import VectorEngine
        engine = VectorEngine(model_name=model_name)
        # Preserve the loaded model revision when provided by transformers.
        first = engine.model._first_module()
        report["resolved_model_revision"] = getattr(getattr(getattr(first, "auto_model", None), "config", None), "_commit_hash", None)
        engine.build_index([record["text"] for record in RECORDS],
                           [{**record, "source": {"snapshot": "original"}} for record in RECORDS])
        for case in CASES:
            report["checks"].append(judge(case, engine.search(case["query"], top_k=3)))
        report["checks"].append({"id": "empty_query", "passed": engine.search("  ") == []})
        unrelated = engine.search("What is the orbital period of Neptune?", top_k=3)
        report["off_topic_observation"] = unrelated
        report["checks"].append({"id": "off_topic_scores_do_not_claim_validated_relevance", "passed": bool(unrelated) and all(row.get("score_semantics") == SCORE_SEMANTICS for row in unrelated)})
        mutable_result = engine.search(CASES[0]["query"], top_k=1)
        mutable_result[0]["source"]["snapshot"] = "tampered"
        report["checks"].append({"id": "returned_metadata_cannot_mutate_index_provenance",
                                 "passed": engine.search(CASES[0]["query"], top_k=1)[0]["source"]["snapshot"] == "original"})
        # Replacing a dataset must not leave the prior employees queryable.
        engine.build_index([], [])
        report["checks"].append({"id": "empty_rebuild_clears_previous_records", "passed": not engine.is_initialized() and engine.search("Python") == []})
        report["status"] = "passed" if all(check["passed"] for check in report["checks"]) else "failed"
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    report["passed"] = sum(check["passed"] for check in report["checks"])
    report["total"] = len(report["checks"])
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate(args.model)
    serialized = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
