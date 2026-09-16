"""Cycle 025 recorded-assessment and aggregate succession contracts."""

import asyncio
from types import SimpleNamespace

import numpy as np
import pandas as pd


def staff(n=12):
    return pd.DataFrame({
        'EmployeeID': [f'E{i}' for i in range(n)],
        'Dept': 'A',
        'Tenure': 2.0,
        'Salary': 100.0,
        'LastRating': 4.0,
        'Age': 30,
        'Attrition': 0,
    })


def test_missing_ratings_remain_unassessed_in_the_aggregate_matrix():
    from src.succession_engine import SuccessionEngine

    frame = staff()
    frame['LastRating'] = np.nan
    result = SuccessionEngine(frame).get_9box_matrix()

    assert set(result['NineBox']) == {'Unassessed'}


def test_summary_preserves_recorded_counts_and_gap_counts():
    from api.routes.succession import get_succession_summary

    engine = SimpleNamespace(
        df=staff(),
        analyze_all=lambda: {
            'readiness': pd.DataFrame({'ReadinessLevel': ['Ready Now', 'Developing']}),
            'gaps': pd.DataFrame({'Dept': ['A']}),
            'bench_strength': pd.DataFrame(),
            'nine_box_summary': pd.DataFrame(),
        },
    )
    result = asyncio.run(get_succession_summary(state=SimpleNamespace(succession_engine=engine)))

    assert result['aggregate_ready_now_count'] == 1
    assert result['critical_gap_count'] == 1
