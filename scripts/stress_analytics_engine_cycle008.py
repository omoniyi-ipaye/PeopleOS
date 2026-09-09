#!/usr/bin/env python3
"""High-volume deterministic stress harness for AnalyticsEngine forensic cycle 008."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.analytics_engine import AnalyticsEngine


def build(rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ids = np.array([f'E{i:07d}' for i in range(rows)], dtype=object)
    dept = rng.choice(['Engineering','Product','Sales','People','Finance','Support',None], rows, p=[.25,.15,.2,.08,.08,.19,.05])
    attrition = rng.choice([0,1,None], rows, p=[.82,.13,.05])
    salary = rng.lognormal(mean=np.log(70_000), sigma=.35, size=rows)
    tenure = rng.gamma(shape=2.0, scale=2.0, size=rows)
    age = np.clip(rng.normal(38, 9, rows), 18, 70)
    rating = rng.choice([1,2,3,4,5,np.nan], rows, p=[.05,.12,.42,.28,.1,.03])
    # Inject invalid/nonfinite rows to exercise exclusions.
    idx = rng.choice(rows, max(20, rows // 500), replace=False)
    salary[idx[:len(idx)//4]] = np.nan
    salary[idx[len(idx)//4:len(idx)//2]] = -1
    tenure[idx[len(idx)//2:3*len(idx)//4]] = -2
    age[idx[3*len(idx)//4:]] = 999
    return pd.DataFrame({'EmployeeID':ids,'Dept':dept,'Salary':salary,'Tenure':tenure,'Age':age,'LastRating':rating,'Attrition':attrition})


def run(rows: int, seed: int) -> dict:
    source = build(rows, seed)
    started = time.perf_counter()
    engine = AnalyticsEngine(source)
    summary = engine.get_summary_statistics()
    departments = engine.get_department_aggregates()
    tenure = engine.get_tenure_distribution()
    ages = engine.get_age_distribution()
    bands = engine.get_salary_bands()
    correlations = engine.get_correlations()
    elapsed = time.perf_counter() - started

    checks = []
    def add(name, ok, actual=None, expected=None):
        checks.append({'name':name,'passed':bool(ok),'actual':actual,'expected':expected})

    add('headcount_department_reconciliation', int(departments['Headcount'].sum()) == summary['headcount'], int(departments['Headcount'].sum()), summary['headcount'])
    add('record_department_reconciliation', int(departments['Total_Records'].sum()) == summary['record_count'], int(departments['Total_Records'].sum()), summary['record_count'])
    add('outcome_department_reconciliation', int(departments['Outcome_Observations'].sum()) == summary['attrition_known_count'], int(departments['Outcome_Observations'].sum()), summary['attrition_known_count'])
    add('tenure_distribution_reconciliation', int(tenure['Count'].sum()) == summary['headcount'], int(tenure['Count'].sum()), summary['headcount'])
    add('age_distribution_reconciliation', int(ages['Count'].sum()) == summary['headcount'], int(ages['Count'].sum()), summary['headcount'])
    add('salary_band_reconciliation', int(bands['Count'].sum()) == summary['salary_observations'], int(bands['Count'].sum()), summary['salary_observations'])
    add('attrition_population_identity', summary['attrition_known_count'] + summary['attrition_excluded_count'] == summary['record_count'])
    add('attrition_share_range', summary['observed_attrition_share'] is None or 0 <= summary['observed_attrition_share'] <= 1)
    add('finite_summary_numbers', all(v is None or not isinstance(v, (float,np.floating)) or math.isfinite(float(v)) for v in summary.values() if not isinstance(v, dict)))
    add('finite_correlation_numbers', correlations.empty or np.isfinite(correlations[['Correlation','P_Value']].to_numpy(dtype=float)).all())
    add('source_rows_unchanged', len(source) == rows, len(source), rows)

    return {
        'cycle':'PEOPLEOS-QUALITY-008', 'engine':'AnalyticsEngine', 'rows':rows, 'seed':seed,
        'elapsed_seconds':round(elapsed,4), 'checks':checks,
        'passed_count':sum(c['passed'] for c in checks), 'check_count':len(checks),
        'passed':all(c['passed'] for c in checks),
        'summary':{'headcount':summary['headcount'],'record_count':summary['record_count'],'salary_observations':summary.get('salary_observations'),'attrition_known_count':summary.get('attrition_known_count')},
    }


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--rows',type=int,default=250000); parser.add_argument('--seed',type=int,default=808); parser.add_argument('--output',default='analytics-engine-cycle008-stress.json')
    args=parser.parse_args(); report=run(args.rows,args.seed)
    Path(args.output).write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(report,indent=2,allow_nan=False))
    if not report['passed']: raise SystemExit(1)

if __name__=='__main__': main()
