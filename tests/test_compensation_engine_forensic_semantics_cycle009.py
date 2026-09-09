"""Additional Cycle 009 semantic challenges for CompensationEngine."""
import numpy as np
import pandas as pd
import pytest

from src.compensation_engine import CompensationEngine


def base_frame(n: int = 40) -> pd.DataFrame:
    return pd.DataFrame({
        'EmployeeID': [f'E{i:04d}' for i in range(n)],
        'Dept': ['Engineering'] * n,
        'JobTitle': ['Common'] * 30 + ['Rare'] * (n - 30),
        'Gender': ['Male', 'Female'] * (n // 2),
        'Tenure': np.arange(n) % 8,
        'Salary': 80_000.0 + np.arange(n) * 1_000.0,
        'LastRating': 3.0,
        'Age': 30.0,
        'Attrition': 0,
    })


def test_external_band_midpoint_is_used_as_true_compa_basis():
    df = base_frame(40)
    df['BandMidpoint'] = 100_000.0
    result = CompensationEngine(df).calculate_compa_ratio()
    assert np.allclose(result['BandMidpoint'], 100_000.0)
    assert np.allclose(result['CompaRatio'], result['Salary'] / 100_000.0)
    assert set(result['MetricSemantics']) == {'supplied_band_midpoint_compa_ratio'}


def test_inconsistent_supplied_ratio_and_midpoint_fail_closed_per_row():
    df = base_frame(40)
    df['BandMidpoint'] = 100_000.0
    df['CompaRatio'] = df['Salary'] / 100_000.0
    df.loc[0, 'CompaRatio'] = 9.99
    result = CompensationEngine(df).calculate_compa_ratio()
    bad = result.loc[result['EmployeeID'].eq('E0000')].iloc[0]
    assert pd.isna(bad['CompaRatio'])
    assert bad['CompaStatus'] == 'Unavailable'
    assert result.loc[~result['EmployeeID'].eq('E0000'), 'CompaRatio'].notna().all()


def test_small_job_title_gender_strata_do_not_expose_exact_suppressed_counts():
    df = base_frame(40)
    # Common gets 15+15; Rare gets 5+5, below default support of 10 per group.
    result = CompensationEngine(df).calculate_gender_pay_gap(10)
    strata = {str(row['job_title']): row for row in result['job_title_strata']}
    assert strata['Common']['eligible'] is True
    assert strata['Common']['male_n'] == 15
    assert strata['Common']['female_n'] == 15
    rare = strata['Rare']
    assert rare['eligible'] is False
    assert rare['suppressed'] is True
    assert rare['male_n'] is None
    assert rare['female_n'] is None
    assert rare['gap_pct'] is None
    assert rare['minimum_group_size'] == 10
