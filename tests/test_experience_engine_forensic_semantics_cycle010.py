"""Second-pass semantic/privacy challenges for ExperienceEngine Cycle 010."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.experience_engine import ExperienceEngine, ExperienceEngineError, MIN_AGGREGATE_SUPPORT


def base_frame(n: int = 40) -> pd.DataFrame:
    idx = np.arange(n)
    return pd.DataFrame({
        'EmployeeID': [f'E{i:04d}' for i in idx],
        'Dept': np.resize(['Engineering', 'Sales'], n),
        'Tenure': np.resize([0.25, 0.75, 1.5, 4.0], n),
        'eNPS_Score': np.resize([0, 2, 4, 6, 8, 10], n).astype(float),
        'Pulse_Score': np.resize([1, 2, 3, 4, 5], n).astype(float),
        'ManagerSatisfaction': np.resize([1, 2, 3, 4, 5], n).astype(float),
        'WorkLifeBalance': np.resize([1, 2, 3, 4, 5], n).astype(float),
        'CareerGrowthSatisfaction': np.resize([1, 2, 3, 4, 5], n).astype(float),
    })


def test_decimal_score_band_boundaries_cover_every_measured_response_exactly_once():
    # Ten observations per score band keeps this arithmetic test outside the
    # privacy-suppression path while exercising every decimal boundary.
    boundary_values = [0.0, 19.9, 20.0, 39.9, 40.0, 59.9, 60.0, 79.9, 80.0, 100.0]
    scores = []
    for value in boundary_values:
        scores.extend([value] * 5)
    df = base_frame(len(scores))
    engine = ExperienceEngine(df)
    engine.df['_exi_score'] = pd.Series(scores, index=engine.df.index)
    engine.respondent_count = len(scores)
    result = engine.get_engagement_segments()
    assert result.get('suppression_applied') is False
    assert sum(int(row['count']) for row in result['segments']) == len(scores)
    assert [row['count'] for row in result['segments']] == [10, 10, 10, 10, 10]


def test_case_insensitive_duplicate_signal_columns_fail_closed_independent_of_order():
    df = base_frame(20)
    df['pulse_score'] = 5.0
    with pytest.raises(ExperienceEngineError, match='Ambiguous case-insensitive'):
        ExperienceEngine(df)
    reversed_columns = df[df.columns[::-1]]
    with pytest.raises(ExperienceEngineError, match='Ambiguous case-insensitive'):
        ExperienceEngine(reversed_columns)


def test_identifier_group_by_does_not_emit_sensitive_group_breakdown():
    df = base_frame(40)
    df['NumericManagerID'] = np.resize(np.arange(4), 40)
    result = ExperienceEngine(df).calculate_experience_index(group_by='NumericManagerID')
    assert 'by_group' not in result or result['by_group'] == []


def test_small_overall_low_score_cell_is_suppressed():
    df = base_frame(40)
    engine = ExperienceEngine(df)
    engine.df['_exi_score'] = 70.0
    engine.df.loc[:3, '_exi_score'] = 10.0
    engine.respondent_count = 40
    result = engine.get_at_risk_employees(threshold=40)
    assert result['available']
    assert result.get('suppressed') is True
    assert result.get('total_at_risk') is None
    assert not result.get('by_department')


def test_score_band_small_cells_use_complementary_suppression():
    df = base_frame(40)
    engine = ExperienceEngine(df)
    engine.df['_exi_score'] = [10.0]*4 + [30.0]*6 + [50.0]*10 + [70.0]*10 + [90.0]*10
    engine.respondent_count = 40
    result = engine.get_engagement_segments()
    rows = {row['segment']: row for row in result['segments']}
    for name in ['Very low score band', 'Low score band']:
        assert rows[name].get('suppressed') is True
        assert rows[name].get('count') is None
        assert rows[name].get('percentage') is None
        assert rows[name].get('avg_exi') is None
    assert rows['Mid score band']['count'] == 10


def test_single_small_score_band_triggers_complementary_suppression_of_second_cell():
    df = base_frame(40)
    engine = ExperienceEngine(df)
    engine.df['_exi_score'] = [10.0]*4 + [50.0]*12 + [70.0]*12 + [90.0]*12
    engine.respondent_count = 40
    result = engine.get_engagement_segments()
    suppressed = [row for row in result['segments'] if row.get('suppressed')]
    assert len(suppressed) >= 2
    assert all(row.get('count') is None and row.get('percentage') is None for row in suppressed)


def test_lifecycle_small_low_score_cell_is_suppressed_inside_supported_stage():
    df = base_frame(40)
    df['Tenure'] = 4.0
    engine = ExperienceEngine(df)
    engine.df['_exi_score'] = 70.0
    engine.df.loc[:3, '_exi_score'] = 10.0
    engine.respondent_count = 40
    result = engine.get_lifecycle_experience()
    stage = result['stages'][0]
    assert stage['respondent_count'] >= MIN_AGGREGATE_SUPPORT
    assert stage.get('at_risk_suppressed') is True
    assert stage.get('at_risk_count') is None


def test_signal_observation_counts_reconcile_component_availability():
    df = base_frame(40)
    df.loc[:9, 'eNPS_Score'] = np.nan
    df.loc[:4, 'Pulse_Score'] = 999
    engine = ExperienceEngine(df)
    assert engine.signal_observation_counts['enps'] == 30
    assert engine.signal_observation_counts['pulse'] == 35
    assert all(0 <= count <= 40 for count in engine.signal_observation_counts.values())
