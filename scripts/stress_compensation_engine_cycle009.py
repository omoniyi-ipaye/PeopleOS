"""High-volume deterministic stress validation for CompensationEngine."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.compensation_engine import CompensationEngine


def build(rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = np.arange(rows)
    attrition = rng.choice([0, 1], rows, p=[0.82, 0.18])
    salary = rng.lognormal(mean=np.log(72_000), sigma=0.38, size=rows)
    salary = np.clip(salary, 18_000, 450_000)
    # Deliberate invalid current salaries and extreme-but-finite valid salaries.
    salary[idx % 997 == 0] = np.nan
    salary[idx % 1237 == 0] = 0
    salary[idx % 50021 == 0] = 1e150
    return pd.DataFrame({
        'EmployeeID': [f'S{i:07d}' for i in idx],
        'Dept': rng.choice(['Engineering','Product','Sales','People','Finance','Ops'], rows),
        'JobTitle': rng.choice(['IC1','IC2','IC3','Lead','Manager'], rows),
        'Gender': rng.choice(['Male','Female','Unknown'], rows, p=[0.47,0.47,0.06]),
        'Tenure': rng.uniform(-0.5, 18, rows),
        'Salary': salary,
        'LastRating': rng.uniform(1, 5, rows),
        'Age': rng.integers(18, 70, rows),
        'Attrition': attrition,
    })


def finite_df(frame: pd.DataFrame, cols: list[str]) -> bool:
    for col in cols:
        if col not in frame:
            continue
        values = pd.to_numeric(frame[col], errors='coerce').dropna().to_numpy(dtype=float)
        if len(values) and not np.isfinite(values).all():
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--rows', type=int, default=250000)
    parser.add_argument('--seed', type=int, default=909)
    parser.add_argument('--output', default='compensation-engine-cycle009-stress.json')
    args = parser.parse_args()

    df = build(args.rows, args.seed)
    source = df.copy(deep=True)
    started = time.perf_counter()
    engine = CompensationEngine(df)
    summary = engine.get_compensation_summary()
    dispersion = engine.calculate_pay_equity_score()
    percentiles = engine.calculate_salary_percentiles()
    bands = engine.get_salary_bands()
    tenure = engine.get_salary_by_tenure()
    gap = engine.calculate_gender_pay_gap()
    association = engine.correlate_salary_with_attrition()
    elapsed = time.perf_counter() - started

    active = source[source['Attrition'].eq(0)]
    active_salary = pd.to_numeric(active['Salary'], errors='coerce')
    valid = active_salary[np.isfinite(active_salary) & (active_salary > 0)]

    checks = []
    def check(name: str, passed: bool, actual=None, expected=None):
        checks.append({'name': name, 'passed': bool(passed), 'actual': actual, 'expected': expected})

    check('salary_observation_reconciliation', summary['salary_observations'] == len(valid), summary['salary_observations'], len(valid))
    check('active_population_reconciliation', summary['active_count'] == len(active), summary['active_count'], len(active))
    check('excluded_salary_reconciliation', summary['excluded_salary_count'] == len(active)-len(valid), summary['excluded_salary_count'], len(active)-len(valid))
    check('salary_band_reconciliation', int(bands['Count'].sum()) == len(valid), int(bands['Count'].sum()), len(valid))
    check('department_percentile_headcount_reconciliation', int(percentiles['Headcount'].sum()) == len(valid), int(percentiles['Headcount'].sum()), len(valid))
    check('department_dispersion_headcount_reconciliation', int(dispersion['Headcount'].sum()) == len(valid), int(dispersion['Headcount'].sum()), len(valid))
    check('tenure_bucket_reconciliation', int(tenure['Count'].sum()) == len(valid), int(tenure['Count'].sum()), len(valid))
    check('finite_summary', all(not isinstance(v, float) or math.isfinite(v) for v in summary.values()))
    check('finite_dispersion', finite_df(dispersion, ['AvgSalary','StdDev','CV','Gini','EquityScore']))
    check('finite_percentiles', finite_df(percentiles, ['P10','P25','P50','P75','P90','Mean','Min','Max']))
    check('gender_gap_finite_or_unavailable', (not gap.get('available')) or (math.isfinite(float(gap['raw_gap_pct'])) and gap.get('p_value') is None or math.isfinite(float(gap['p_value']))))
    check('salary_attrition_finite_or_unavailable', (not association.get('available')) or (math.isfinite(float(association['correlation'])) and math.isfinite(float(association['p_value']))))
    check('source_rows_unchanged', source.equals(df), len(df), len(source))

    report = {
        'cycle': 'PEOPLEOS-QUALITY-009',
        'engine': 'CompensationEngine',
        'rows': args.rows,
        'seed': args.seed,
        'elapsed_seconds': round(elapsed, 4),
        'checks': checks,
        'passed_count': sum(c['passed'] for c in checks),
        'check_count': len(checks),
        'passed': all(c['passed'] for c in checks),
        'summary': {
            'active_count': summary['active_count'],
            'salary_observations': summary['salary_observations'],
            'excluded_salary_count': summary['excluded_salary_count'],
        },
    }
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
