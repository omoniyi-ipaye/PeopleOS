"""Cycle 021 forensic contracts for QualityOfHireEngine.

These tests keep Quality of Hire aggregate-only and make measurement maturity,
denominators, composite comparability and role/source composition explicit.
All people and outcomes are synthetic.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes import quality_of_hire as routes
from src.quality_of_hire_engine import QualityOfHireEngine


def workforce(n: int = 40) -> pd.DataFrame:
    half = n // 2
    return pd.DataFrame({
        'EmployeeID': [f'E{i:04d}' for i in range(n)],
        'Dept': ['Engineering'] * half + ['People'] * (n - half),
        'JobLevel': ['L3'] * half + ['L4'] * (n - half),
        'JobTitle': ['Engineer'] * half + ['Partner'] * (n - half),
        'Tenure': [2.0] * n,
        'LastRating': [4.0] * n,
        'Attrition': [0] * n,
        'HireSource': ['Referral'] * half + ['JobBoard'] * (n - half),
        'InterviewScore': [4.0] * n,
        'HireDate': ['2024-01-01'] * n,
    })


def test_total_hires_and_observed_performance_denominator_are_separate():
    frame = workforce(20)
    frame['HireSource'] = 'Referral'
    frame.loc[4:, 'LastRating'] = np.nan

    row = QualityOfHireEngine(frame).calculate_source_effectiveness().set_index('HireSource').loc['Referral']

    assert row['total_hires'] == 20
    assert row['performance_recorded_observations'] == 4
    assert row['performance_observations'] == 4
    assert row['performance_coverage'] == 0.2
    assert pd.isna(row['quality_score'])
    assert row['grade'] == 'Unavailable'
    assert 'performance' in row['quality_unavailable_reason']


def test_retention_requires_configured_duration_exposure():
    frame = workforce(20)
    frame['HireSource'] = 'Referral'
    frame['Tenure'] = 0.5

    row = QualityOfHireEngine(frame).calculate_source_effectiveness().set_index('HireSource').loc['Referral']

    assert row['retention_recorded_observations'] == 20
    assert row['retention_eligible_hires'] == 0
    assert row['retention_observations'] == 0
    assert row['retention_rate'] is None
    assert row['retention_maturity'] == 'insufficient_12mo_exposure'
    assert pd.isna(row['quality_score'])


def test_source_composites_use_one_effective_construct_and_suppress_missing_cohorts():
    frame = workforce()
    frame.loc[20:, 'Attrition'] = np.nan

    rows = QualityOfHireEngine(frame).calculate_source_effectiveness().set_index('HireSource')
    measured, missing = rows.loc['Referral'], rows.loc['JobBoard']

    assert measured['quality_score'] is not None and np.isfinite(measured['quality_score'])
    assert pd.isna(missing['quality_score'])
    assert measured['effective_quality_weights'] == missing['effective_quality_weights']
    assert measured['effective_quality_weights'] == pytest.approx({'performance': 4 / 7, 'retention': 3 / 7})
    assert 'promotion' in measured['excluded_quality_components']
    assert measured['quality_claim'] == 'descriptive_observed_composite_not_hiring_effectiveness'


def test_role_mix_is_aggregate_and_source_specific():
    rows = QualityOfHireEngine(workforce()).calculate_source_effectiveness().set_index('HireSource')

    assert rows.loc['Referral', 'role_mix']['Dept'] == {'Engineering': 1.0}
    assert rows.loc['JobBoard', 'role_mix']['Dept'] == {'People': 1.0}
    assert rows.loc['Referral', 'role_mix_columns'] == ['Dept', 'JobLevel', 'JobTitle']


def test_insufficient_prehire_support_is_reported_as_a_measurement_gap():
    frame = workforce()
    frame.loc[5:, 'InterviewScore'] = np.nan

    result = QualityOfHireEngine(frame).correlate_prehire_posthire()
    gap = next(item for item in result['measurement_gaps'] if item['predictor'] == 'InterviewScore')

    assert result['outcome_observations'] == 40
    assert result['outcome_maturity'] == 'duration_qualified_observed'
    assert result['correlations'] == []
    assert gap['paired_observations'] == 5
    assert gap['minimum_paired_observations'] == 20
    assert gap['reason'] == 'insufficient_paired_observations'


def test_source_score_claim_boundary_survives_full_analysis_and_api():
    engine = QualityOfHireEngine(workforce())
    state = SimpleNamespace(quality_of_hire_engine=engine, raw_df=engine.df)
    app = FastAPI()
    app.dependency_overrides[routes.require_quality_of_hire] = lambda: state
    app.include_router(routes.router)

    with TestClient(app) as client:
        response = client.get('/api/quality-of-hire/analysis')

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload['summary']['best_source_semantics'] == 'descriptive_composite_only_not_hiring_effectiveness'
    assert payload['source_effectiveness'][0]['total_hires'] == 20
    assert payload['source_effectiveness'][0]['performance_recorded_observations'] == 20
    assert payload['source_effectiveness'][0]['effective_quality_weights']
    assert payload['source_effectiveness'][0]['role_mix']['Dept']
    assert any('prospective' in warning.lower() for warning in payload['warnings'])


def test_engine_does_not_mutate_input_when_normalizing_or_reporting():
    frame = workforce()
    original = frame.copy(deep=True)

    engine = QualityOfHireEngine(frame)
    engine.analyze_all()

    pd.testing.assert_frame_equal(frame, original)


def test_string_encoded_attrition_is_normalized_before_retention_aggregation():
    frame = workforce(20)
    frame['HireSource'] = 'Referral'
    frame['Attrition'] = ['0'] * 20

    row = QualityOfHireEngine(frame).calculate_source_effectiveness().iloc[0]

    assert row['retention_observations'] == 20
    assert row['attrition_count'] == 0
    assert row['retention_rate'] == 1.0
