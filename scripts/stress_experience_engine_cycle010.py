"""Deterministic 250k-row stress harness for ExperienceEngine Cycle 010."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.experience_engine import ExperienceEngine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--rows', type=int, default=250000)
    parser.add_argument('--output', default='experience-engine-cycle010-stress.json')
    args = parser.parse_args()

    rng = np.random.default_rng(1010)
    n = args.rows
    idx = np.arange(n)
    df = pd.DataFrame({
        'EmployeeID': [f'E{i:07d}' for i in idx],
        'Dept': rng.choice(['Engineering','Product','Sales','People','Finance','Support'], n),
        'ManagerID': [f'M{v:05d}' for v in rng.integers(0, 5000, n)],
        'NumericManagerID': rng.integers(0, 5000, n),
        'Tenure': rng.uniform(0, 18, n),
        'eNPS_Score': rng.integers(0, 11, n).astype(float),
        'Pulse_Score': rng.integers(1, 6, n).astype(float),
        'ManagerSatisfaction': rng.integers(1, 6, n).astype(float),
        'WorkLifeBalance': rng.integers(1, 6, n).astype(float),
        'CareerGrowthSatisfaction': rng.integers(1, 6, n).astype(float),
    })
    df.loc[idx % 37 == 0, 'eNPS_Score'] = np.nan
    df.loc[idx % 53 == 0, 'Pulse_Score'] = 999
    df.loc[idx % 71 == 0, 'ManagerSatisfaction'] = np.nan
    original = df.copy(deep=True)

    started = time.perf_counter()
    engine = ExperienceEngine(df)
    index = engine.calculate_experience_index(group_by='Dept')
    segments = engine.get_engagement_segments()
    drivers = engine.identify_experience_drivers()
    low_score = engine.get_at_risk_employees()
    lifecycle = engine.get_lifecycle_experience()
    elapsed = time.perf_counter() - started

    checks = []
    def check(name, passed, actual=None, expected=None):
        checks.append({'name': name, 'passed': bool(passed), 'actual': actual, 'expected': expected})

    measured = int(engine.df['_exi_score'].notna().sum())
    check('respondent_reconciliation', index.get('respondent_count') == measured, index.get('respondent_count'), measured)
    check('coverage_reconciliation', abs(index.get('response_coverage', 0) - measured / n) < 1e-12)
    check('segment_count_reconciliation', sum(int(r['count']) for r in segments['segments']) == measured)
    check('segment_percentage_reconciliation', abs(sum(float(r['percentage']) for r in segments['segments']) - 100.0) <= 0.2)
    check('department_groups_supported', all(int(r['count']) >= 10 for r in index.get('by_group', [])))
    factors = [str(r['factor']).lower() for r in drivers.get('drivers', [])]
    check('identifier_exclusion', not any(f == 'id' or f.endswith('id') or '_id' in f for f in factors))
    check('driver_finiteness', all(np.isfinite(float(r['correlation'])) for r in drivers.get('drivers', [])))
    check('aggregate_low_score_privacy', low_score.get('employees') in (None, []) and 'EmployeeID' not in repr(low_score))
    check('low_score_department_support', all(int(r['at_risk_count']) >= 10 for r in low_score.get('by_department', [])))
    check('lifecycle_support', all(int(r['respondent_count']) >= 10 for r in lifecycle.get('stages', [])))
    check('score_finiteness', np.isfinite(engine.df['_exi_score'].dropna().to_numpy(dtype=float)).all())
    check('score_range', engine.df['_exi_score'].dropna().between(0, 100).all())
    check('manager_ranking_disabled', engine.analyze_manager_impact().get('available') is False)
    check('source_rows_unchanged', df.equals(original), len(df), len(original))

    report = {
        'cycle': 'PEOPLEOS-QUALITY-010',
        'engine': 'ExperienceEngine',
        'rows': n,
        'seed': 1010,
        'elapsed_seconds': round(elapsed, 4),
        'checks': checks,
        'passed_count': sum(1 for c in checks if c['passed']),
        'check_count': len(checks),
        'passed': all(c['passed'] for c in checks),
        'summary': {
            'respondent_count': measured,
            'response_coverage': index.get('response_coverage'),
            'signals_available': index.get('signals_available'),
        },
    }
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
