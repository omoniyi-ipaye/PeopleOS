#!/usr/bin/env python3
"""Deterministic, known-answer synthetic workforce and stress validation.

The generated records are fictional.  This validates software mechanics and
measurement contracts, not organizational construct validity or future model
performance.  No generated fixture is a final model holdout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import time
import sys
import socket
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analytics_engine import AnalyticsEngine
from src.compensation_engine import CompensationEngine, CompensationEngineError
from src.platform.local_dataset_store import load_dataset_artifact, save_dataset_artifact
from src.platform.workspace import WorkspaceStore
from src.sentiment_engine import SentimentEngine

SEED = 20260908
DEPARTMENTS = ("Engineering", "Sales", "Operations", "People", "Tiny")
LOCATIONS = ("Madrid", "Berlin", "London", "Lagos")


def synthetic_workforce(rows: int, *, seed: int = SEED, drift: bool = False) -> pd.DataFrame:
    """Create a reproducible current-state workforce with deliberate edge cases."""
    if rows < 100:
        raise ValueError("Stress fixtures require at least 100 rows")
    rng = np.random.default_rng(seed)
    employee_id = np.array([f"SYN-{i:07d}" for i in range(rows)], dtype=object)
    dept = rng.choice(DEPARTMENTS[:-1], rows, p=[.40, .25, .25, .10]).astype(object)
    dept[:7] = "Tiny"  # always below the governed reporting threshold of 10
    gender = rng.choice(["Female", "Male", "Non-binary", "Unknown"], rows,
                        p=[.48, .47, .02, .03])
    location = rng.choice(LOCATIONS, rows, p=[.45, .20, .20, .15])
    age = rng.integers(18, 70, rows).astype(float)
    tenure = np.round(rng.uniform(0, 25, rows), 2)
    rating = np.round(np.clip(rng.normal(3.4, .8, rows), 1, 5), 1)
    base = 36000 + tenure * 1800 + (dept == "Engineering") * 16000 + (dept == "Sales") * 7000
    salary = np.round(base + rng.normal(0, 6500, rows), 2)
    # A known, non-causal outcome rule. Drift changes prevalence and feature relation.
    score = -.8 + .6 * (dept == "Sales") + .5 * (tenure < 1) - .35 * (rating >= 4)
    if drift:
        score = score + 1.0 + .6 * (dept == "Operations") - .6 * (dept == "Sales")
        salary *= 1.08
    probability = 1 / (1 + np.exp(-score))
    attrition = (rng.random(rows) < probability).astype(float)
    manager = np.array(["CEO" if i < 12 else employee_id[(i - 12) // 8] for i in range(rows)], dtype=object)
    hire_dates = pd.Timestamp("2026-09-01") - pd.to_timedelta((tenure * 365.25).astype(int), unit="D")
    frame = pd.DataFrame({
        "EmployeeID": employee_id, "Gender": gender, "Age": age, "Dept": dept,
        "JobTitle": np.where(dept == "Engineering", "Engineer", "Specialist"),
        "JobLevel": rng.choice(["L1", "L2", "L3", "L4", "L5"], rows),
        "Tenure": tenure, "Location": location, "Country": np.where(location == "Madrid", "Spain", "Other"),
        "Salary": salary, "LastRating": rating, "ManagerID": manager,
        "Attrition": attrition, "HireSource": rng.choice(["Referral", "Direct", "Agency"], rows),
        "HireDate": hire_dates.strftime("%Y-%m-%d"), "SnapshotDate": "2026-09-01",
        "PromotionCount": rng.integers(0, 4, rows), "InterviewScore": np.round(rng.uniform(1, 5, rows), 1),
        "AssessmentScore": np.round(rng.uniform(0, 100, rows), 1),
    })
    # Measured missingness and unknown employment status must not become zero.
    frame.loc[frame.index % 17 == 0, "Salary"] = np.nan
    frame.loc[frame.index % 19 == 0, "LastRating"] = np.nan
    frame.loc[frame.index % 23 == 0, "Attrition"] = np.nan
    return frame


def synthetic_survey(workforce: pd.DataFrame, *, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1)
    sample = workforce.iloc[::2][["EmployeeID", "Dept"]].copy()
    sample["SurveyDate"] = "2026-08-15"
    sample["eNPSScore"] = rng.integers(0, 11, len(sample))
    sample["SentimentScore"] = np.round(rng.uniform(0, 1, len(sample)), 4)
    sample["SentimentLabel"] = np.select(
        [sample["SentimentScore"] >= .6, sample["SentimentScore"] <= .4],
        ["positive", "negative"], default="neutral")
    return sample


def _finite(value) -> bool:
    if isinstance(value, dict):
        return all(_finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(v) for v in value)
    return not isinstance(value, (float, np.floating)) or math.isfinite(float(value))


def live_api_stress(frame: pd.DataFrame, *, workers: int = 8) -> dict:
    """Upload the large fixture through TCP and challenge concurrent read consistency."""
    with tempfile.TemporaryDirectory(prefix="peopleos-live-stress-") as home:
        env = {**os.environ, "PEOPLEOS_HOME": home,
               "PEOPLEOS_WORKSPACE_REGISTRY": str(Path(home) / "workspace.json"),
               "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", str(port)],
            cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        base = f"http://127.0.0.1:{port}"
        try:
            with httpx.Client(base_url=base, timeout=180, trust_env=False) as client:
                for _ in range(240):
                    if proc.poll() is not None:
                        raise RuntimeError("Stress API exited before becoming healthy")
                    try:
                        if client.get("/").status_code == 200:
                            break
                    except httpx.ConnectError:
                        pass
                    time.sleep(.25)
                else:
                    raise RuntimeError("Stress API did not become healthy")
                content = frame.to_csv(index=False).encode()
                upload_started = time.perf_counter()
                response = client.post("/api/upload", files={"file": ("synthetic-stress.csv", content, "text/csv")})
                upload_seconds = time.perf_counter() - upload_started
                response.raise_for_status()
                uploaded = response.json()
                dataset_id = uploaded["dataset_id"]

            request_count = max(32, workers * 4)
            def read_summary(_):
                started = time.perf_counter()
                with httpx.Client(base_url=base, timeout=60, trust_env=False) as client:
                    response = client.get("/api/analytics/summary")
                return response.status_code, response.headers.get("x-peopleos-dataset"), response.json(), time.perf_counter() - started
            with ThreadPoolExecutor(max_workers=workers) as pool:
                reads = list(pool.map(read_summary, range(request_count)))
            latencies = sorted(item[3] for item in reads)
            return {
                "rows_uploaded": uploaded["rows_loaded"], "dataset_id": dataset_id,
                "upload_seconds": upload_seconds, "concurrent_reads": request_count,
                "successful_reads": sum(item[0] == 200 for item in reads),
                "consistent_dataset_headers": all(item[1] == dataset_id for item in reads),
                "consistent_bodies": all(item[2] == reads[0][2] for item in reads),
                "summary": reads[0][2], "p95_read_seconds": latencies[max(0, math.ceil(.95 * len(latencies)) - 1)],
            }
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()


def run(*, rows: int = 20_000, seed: int = SEED, workers: int = 8, live_api: bool = False) -> dict:
    started = time.perf_counter()
    checks: list[dict] = []

    def check(name: str, actual, expected, *, detail: str | None = None):
        if isinstance(actual, np.generic):
            actual = actual.item()
        if isinstance(expected, np.generic):
            expected = expected.item()
        passed = bool(np.isclose(actual, expected, equal_nan=True)) if (
            isinstance(actual, (int, float, np.number)) and isinstance(expected, (int, float, np.number))
        ) else bool(actual == expected)
        checks.append({"name": name, "actual": actual, "expected": expected,
                       "passed": passed, **({"detail": detail} if detail else {})})

    frame = synthetic_workforce(rows, seed=seed)
    generation_seconds = time.perf_counter() - started
    expected_current = frame.loc[frame["Attrition"].fillna(1).eq(0)]
    expected_known = frame["Attrition"].notna()
    analytics_start = time.perf_counter()
    summary = AnalyticsEngine(frame).get_summary_statistics()
    compensation = CompensationEngine(frame).get_compensation_summary()
    analytics_seconds = time.perf_counter() - analytics_start
    check("unique synthetic identities", frame["EmployeeID"].nunique(), rows)
    check("record count", summary["record_count"], rows)
    check("active headcount", summary["active_count"], int((frame["Attrition"] == 0).sum()))
    check("known outcome denominator", summary["attrition_known_count"], int(expected_known.sum()))
    check("observed attrition share", summary["observed_attrition_share"], float(frame.loc[expected_known, "Attrition"].mean()))
    check("active salary observations", summary["salary_observations"], int(expected_current["Salary"].notna().sum()))
    check("active payroll", compensation["total_payroll"], float(expected_current["Salary"].sum()))
    check("all analytics JSON values finite", _finite(summary) and _finite(compensation), True)

    departments = AnalyticsEngine(frame).get_department_aggregates()
    check("department records reconcile", int(departments["Total_Records"].sum()), rows)
    check("tiny group deliberately present", int((frame["Dept"] == "Tiny").sum()), 7)

    # Row order, display precision and extra unused columns must not change answers.
    transformed = frame.sample(frac=1, random_state=seed).reset_index(drop=True)
    transformed["UnusedHRISExportColumn"] = "ignore-me"
    invariant = AnalyticsEngine(transformed).get_summary_statistics()
    for key in ("record_count", "active_count", "attrition_known_count", "observed_attrition_share",
                "salary_mean", "salary_median", "salary_observations"):
        check(f"permutation invariance: {key}", invariant[key], summary[key])

    survey = synthetic_survey(frame, seed=seed)
    sentiment = SentimentEngine(frame, enps_df=survey)
    enps = sentiment.calculate_enps()
    promoters = int((survey["eNPSScore"] >= 9).sum())
    detractors = int((survey["eNPSScore"] <= 6).sum())
    check("survey response denominator", enps["total_responses"], len(survey))
    check("independent eNPS arithmetic", enps["overall_enps"], round((promoters - detractors) / len(survey) * 100, 1))

    drifted = synthetic_workforce(rows, seed=seed, drift=True)
    drift_summary = AnalyticsEngine(drifted).get_summary_statistics()
    check("drift fixture changes outcome prevalence",
          abs(drift_summary["observed_attrition_share"] - summary["observed_attrition_share"]) > .10, True)
    check("drift fixture changes pay distribution", drift_summary["salary_mean"] > summary["salary_mean"], True)

    malformed = frame.copy()
    malformed.loc[1, "EmployeeID"] = malformed.loc[0, "EmployeeID"]
    check("malformed duplicate identity exists", malformed["EmployeeID"].duplicated().any(), True)
    extreme = frame.head(100).copy()
    extreme["Attrition"] = 0
    extreme["Salary"] = 1e308
    try:
        CompensationEngine(extreme).get_compensation_summary()
        extreme_failed_closed = False
    except (CompensationEngineError, ValueError, OverflowError):
        extreme_failed_closed = True
    check("unrepresentable payroll fails closed", extreme_failed_closed, True)

    # Exercise durable artifacts and concurrent registry writes in an isolated home.
    with tempfile.TemporaryDirectory(prefix="peopleos-synthetic-stress-") as home:
        prior_home = os.environ.get("PEOPLEOS_HOME")
        os.environ["PEOPLEOS_HOME"] = home
        try:
            artifact = frame.head(min(rows, 10_000))
            save_dataset_artifact("stress-dataset", artifact)
            restored = load_dataset_artifact("stress-dataset")
            check("artifact row-count round trip", len(restored), len(artifact))
            check("artifact identity round trip",
                  restored["EmployeeID"].tolist() == artifact["EmployeeID"].tolist(), True)
            registry = Path(home) / "workspace.json"
            stores = [WorkspaceStore(registry) for _ in range(workers)]
            initial_workspaces = len(WorkspaceStore(registry).list_workspaces())
            writes = max(64, workers * 8)
            def create(index: int):
                stores[index % workers].ensure_workspace(f"stress-{index:04d}")
            with ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(create, range(writes)))
            final_workspaces = WorkspaceStore(registry).list_workspaces()
            check("concurrent registry writes preserved", len(final_workspaces), initial_workspaces + writes)
            check("concurrent identities remain unique",
                  len({w.workspace_id for w in final_workspaces}), len(final_workspaces))
        finally:
            if prior_home is None:
                os.environ.pop("PEOPLEOS_HOME", None)
            else:
                os.environ["PEOPLEOS_HOME"] = prior_home

    api_result = None
    if live_api:
        api_result = live_api_stress(frame, workers=workers)
        check("live API accepted all rows", api_result["rows_uploaded"], rows)
        check("live API concurrent reads succeeded", api_result["successful_reads"], api_result["concurrent_reads"])
        check("live API dataset headers remain consistent", api_result["consistent_dataset_headers"], True)
        check("live API concurrent bodies remain consistent", api_result["consistent_bodies"], True)
        check("live API active headcount", api_result["summary"]["active_count"], int((frame["Attrition"] == 0).sum()))
        check("live API completes within bounded validation timeout", api_result["upload_seconds"] < 120, True)
        check("live API p95 concurrent read latency below 10 seconds", api_result["p95_read_seconds"] < 10, True)
    elapsed = time.perf_counter() - started
    return {
        "schema_version": 1,
        "scope": "fictional_known_answer_software_and_load_validation_not_real_workforce_or_predictive_certification",
        "seed": seed, "rows": rows, "workers": workers,
        "fixture_sha256": hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest(),
        "profiles": {
            "baseline": {"known_outcomes": int(expected_known.sum()), "active": int((frame["Attrition"] == 0).sum())},
            "missingness": {"salary": int(frame["Salary"].isna().sum()), "rating": int(frame["LastRating"].isna().sum()),
                            "employment_status": int(frame["Attrition"].isna().sum())},
            "privacy": {"tiny_group_records": 7, "reporting_minimum": 10},
            "drift": {"baseline_attrition_share": summary["observed_attrition_share"],
                      "shifted_attrition_share": drift_summary["observed_attrition_share"]},
            "adversarial": ["duplicate identity", "unrepresentable aggregate payroll", "unused HRIS column", "row permutation"],
        },
        "timings_seconds": {"generation": generation_seconds, "analytics": analytics_seconds, "total": elapsed},
        "live_api": api_result,
        "checks": checks, "passed": sum(c["passed"] for c in checks),
        "failed": sum(not c["passed"] for c in checks), "complete": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--live-api", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/validation/synthetic-stress-results.json")
    args = parser.parse_args()
    result = run(rows=args.rows, seed=args.seed, workers=args.workers, live_api=args.live_api)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"RESULT: {result['passed']} passed; {result['failed']} failed; {result['rows']} fictional employees")
    print(f"REPORT: {args.output}")
    return int(result["failed"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
