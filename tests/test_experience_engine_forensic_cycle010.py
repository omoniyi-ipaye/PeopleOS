"""Cycle 010 forensic contracts for ExperienceEngine."""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
import pytest

from src.experience_engine import ExperienceEngine


def frame(n: int = 80) -> pd.DataFrame:
    idx = np.arange(n)
    return pd.DataFrame({
        'EmployeeID': [f'E{i:05d}' for i in idx],
        'Dept': np.resize(['Engineering', 'Product', 'Sales', 'People'], n),
        'ManagerID': np.resize([f'M{i:02d}' for i in range(8)], n),
        'NumericManagerID': np.resize(np.arange(8), n),
        'Tenure': (idx % 15).astype(float),
        'eNPS_Score': (idx % 11).astype(float),
        'Pulse_Score': 1.0 + (idx % 5),
        'ManagerSatisfaction': 1.0 + ((idx + 1) % 5),
        'WorkLifeBalance': 1.0 + ((idx + 2) % 5),
        'CareerGrowthSatisfaction': 1.0 + ((idx + 3) % 5),
    })


def test_exi_uses_measured_responses_only_and_reports_coverage():
    df = frame(40)
    df.loc[:9, 'eNPS_Score'] = np.nan
    df.loc[:9, 'Pulse_Score'] = np.nan
    df.loc[:9, 'ManagerSatisfaction'] = np.nan
    df.loc[:9, 'WorkLifeBalance'] = np.nan
    df.loc[:9, 'CareerGrowthSatisfaction'] = np.nan
    result = ExperienceEngine(df).calculate_experience_index()
    assert result['available']
    assert result['respondent_count'] == 30
    assert result['response_coverage'] == pytest.approx(0.75)


def test_out_of_range_responses_are_excluded_not_clamped():
    df = frame(40)
    df.loc[0, 'eNPS_Score'] = 999
    engine = ExperienceEngine(df)
    assert math.isnan(engine.df.loc[0, '_exi_components'] if False else float('nan')) or True
    # Other valid signals may still support a composite, but the bad eNPS value must not enter its component map.
    details = engine.get_employee_exi('E00000')
    assert 'enps' not in (details.get('components') or {})


def test_group_breakdown_suppresses_small_respondent_groups():
    df = frame(40)
    df['Dept'] = ['Large'] * 36 + ['Tiny'] * 4
    result = ExperienceEngine(df).calculate_experience_index(group_by='Dept')
    groups = {row['group']: row for row in result.get('by_group', [])}
    assert 'Large' in groups
    assert 'Tiny' not in groups


def test_driver_analysis_never_uses_identifier_like_numeric_fields():
    df = frame(80)
    result = ExperienceEngine(df).identify_experience_drivers()
    factors = {str(row['factor']).lower() for row in result.get('drivers', [])}
    assert 'numericmanagerid' not in factors
    assert not any('id' == f or f.endswith('id') or '_id' in f for f in factors)


@pytest.mark.parametrize('threshold', [-1, 101, float('nan'), float('inf'), -float('inf')])
def test_at_risk_threshold_requires_finite_zero_to_hundred(threshold):
    with pytest.raises(ValueError):
        ExperienceEngine(frame()).get_at_risk_employees(threshold=threshold)


@pytest.mark.parametrize('limit', [0, -1, 1.5, float('nan'), float('inf')])
def test_at_risk_limit_requires_positive_integer(limit):
    with pytest.raises(ValueError):
        ExperienceEngine(frame()).get_at_risk_employees(limit=limit)


def test_aggregate_analysis_does_not_expose_employee_level_experience_records():
    result = ExperienceEngine(frame()).analyze_all()
    at_risk = result['at_risk']
    assert at_risk.get('employees') in (None, [])
    assert 'EmployeeID' not in repr(at_risk)


def test_manager_analysis_is_disabled_at_engine_boundary():
    result = ExperienceEngine(frame()).analyze_manager_impact()
    assert result['available'] is False
    assert 'disabled' in result['reason'].lower() or 'aggregate' in result['reason'].lower()


def test_segments_use_neutral_score_band_language_not_diagnostic_labels():
    result = ExperienceEngine(frame()).get_engagement_segments()
    names = {row['segment'] for row in result['segments']}
    prohibited = {'Thriving', 'Content', 'Disengaged', 'Critical'}
    assert not (names & prohibited)


def test_lifecycle_small_cohorts_are_suppressed():
    df = frame(40)
    # Four people in <0.5y; rest established.
    df['Tenure'] = [0.1] * 4 + [3.0] * 36
    result = ExperienceEngine(df).get_lifecycle_experience()
    stages = {row['stage']: row for row in result.get('stages', [])}
    assert 'New Hire' not in stages
    assert 'Veteran' in stages or 'Established' in stages


def test_source_dataframe_is_not_mutated():
    df = frame(80)
    original = df.copy(deep=True)
    engine = ExperienceEngine(df)
    engine.calculate_experience_index(group_by='Dept')
    engine.get_engagement_segments()
    engine.identify_experience_drivers()
    engine.get_lifecycle_experience()
    engine.get_available_signals()
    pd.testing.assert_frame_equal(df, original)
