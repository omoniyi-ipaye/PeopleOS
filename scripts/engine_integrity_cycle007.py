#!/usr/bin/env python3
"""Independent numeric oracle harness for PeopleOS Quality Cycle 007.

The expected values in this script are calculated directly from controlled source
rows with plain Python arithmetic. They intentionally do not reuse PeopleOS
population/analytics helper functions to construct expected results.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.agent.analysis_sandbox import AnalysisSpec, GovernedAnalysisSandbox
from src.analytics_engine import AnalyticsEngine
from src.compensation_engine import CompensationEngine


def source_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i in range(24):
        attrition: int | None = 1 if i in {18, 19, 20, 21} else 0
        if i in {22, 23}:
            attrition = None
        rows.append({
            'EmployeeID': f'E{i:03d}',
            'Dept': 'Engineering' if i < 12 else 'Product',
            'Location': 'Madrid' if i % 2 == 0 else 'London',
            'JobLevel': 'L3' if i % 3 else 'L4',
            'JobTitle': 'Analyst',
            'Gender': 'Female' if i % 2 else 'Male',
            'Salary': float(60_000 + i * 1_000),
            'Tenure': float(i % 6),
            'Age': float(26 + i),
            'LastRating': float(1 + i % 5),
            'Attrition': attrition,
            'SnapshotDate': '2026-09-01',
        })
    return rows


def expected(rows: list[dict[str, Any]]) -> dict[str, Any]:
    known = [row for row in rows if row['Attrition'] in (0, 1)]
    active = [row for row in rows if row['Attrition'] == 0]
    salary = [float(row['Salary']) for row in active if row['Salary'] is not None and float(row['Salary']) > 0]
    departments = sorted({str(row['Dept']) for row in active})

    by_location: dict[str, list[float]] = {}
    for row in active:
        by_location.setdefault(str(row['Location']), []).append(float(row['Salary']))

    return {
        'record_count': len(rows),
        'headcount': len(active),
        'known_outcomes': len(known),
        'departures': sum(int(row['Attrition']) for row in known),
        'observed_attrition_share': sum(int(row['Attrition']) for row in known) / len(known),
        'salary_observations': len(salary),
        'salary_mean': sum(salary) / len(salary),
        'total_payroll': sum(salary),
        'department_count': len(departments),
        'location_salary_sum': {key: sum(values) for key, values in by_location.items()},
    }


def check(name: str, actual: Any, wanted: Any, tolerance: float = 1e-9) -> dict[str, Any]:
    if isinstance(wanted, float):
        passed = actual is not None and abs(float(actual) - wanted) <= max(tolerance, abs(wanted) * 1e-12)
    else:
        passed = actual == wanted
    return {'name': name, 'passed': bool(passed), 'actual': actual, 'expected': wanted}


def run() -> dict[str, Any]:
    rows = source_rows()
    oracle = expected(rows)
    frame = pd.DataFrame(rows)

    analytics_engine = AnalyticsEngine(frame)
    analytics = analytics_engine.get_summary_statistics()
    compensation = CompensationEngine(frame).get_compensation_summary()
    department = analytics_engine.get_department_aggregates()

    comparison = GovernedAnalysisSandbox(frame).run(AnalysisSpec(
        operation='compare_groups', population='active', group_by='Location', measure='Salary',
        statistic='sum', group_a='Madrid', group_b='London',
    ))
    compare_values = {item['group']: item['value'] for item in comparison.get('output', {}).get('groups', [])} if comparison.get('available') else {}

    checks = [
        check('analytics.record_count', analytics['record_count'], oracle['record_count']),
        check('analytics.headcount', analytics['headcount'], oracle['headcount']),
        check('analytics.attrition_known_count', analytics['attrition_known_count'], oracle['known_outcomes']),
        check('analytics.attrition_count', analytics['attrition_count'], oracle['departures']),
        check('analytics.observed_attrition_share', analytics['observed_attrition_share'], oracle['observed_attrition_share']),
        check('analytics.salary_observations', analytics['salary_observations'], oracle['salary_observations']),
        check('analytics.salary_mean', analytics['salary_mean'], oracle['salary_mean']),
        check('analytics.department_count', analytics['department_count'], oracle['department_count']),
        check('compensation.headcount', compensation['headcount'], oracle['salary_observations']),
        check('compensation.avg_salary', compensation['avg_salary'], oracle['salary_mean']),
        check('compensation.total_payroll', compensation['total_payroll'], oracle['total_payroll']),
        check('departments.headcount_reconciliation', int(department['Headcount'].sum()), oracle['headcount']),
        check('departments.records_reconciliation', int(department['Total_Records'].sum()), oracle['record_count']),
        check('departments.outcomes_reconciliation', int(department['Outcome_Observations'].sum()), oracle['known_outcomes']),
        check('sandbox.sum.madrid', compare_values.get('Madrid'), oracle['location_salary_sum']['Madrid']),
        check('sandbox.sum.london', compare_values.get('London'), oracle['location_salary_sum']['London']),
    ]

    return {
        'schema_version': 1,
        'cycle': 'PEOPLEOS-QUALITY-007',
        'fixture': 'independent-controlled-24-employee-oracle',
        'oracle': oracle,
        'checks': checks,
        'passed': all(item['passed'] for item in checks),
        'passed_count': sum(item['passed'] for item in checks),
        'check_count': len(checks),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='engine-integrity-cycle007.json')
    args = parser.parse_args()
    report = run()
    Path(args.output).write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report, indent=2, allow_nan=False))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
