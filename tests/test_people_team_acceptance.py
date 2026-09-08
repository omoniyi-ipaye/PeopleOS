"""Independent People-team acceptance cases with explicit source denominators.

These use synthetic measurements with hand-computable answers. They verify
software behavior, not prospective model accuracy on a real workforce.
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.analytics_engine import AnalyticsEngine
from src.compensation_engine import CompensationEngine
from src.ml_engine import MLEngineError
from src.model_training import binary_metrics


def workforce():
    return pd.DataFrame({
        'EmployeeID': [f'{i:04d}' for i in range(8)],
        'Dept': ['People', ' People ', None, '', 'Finance', 'Finance', 'People', 'People'],
        'Salary': [60000., 80000., 40000., 100000., None, -1., 900000., 800000.],
        'Tenure': [0., 2., None, -1., 1., 1., 3., 4.],
        'Attrition': [0, 0, 0, 0, 0, 0, 1, None],
    })


@pytest.fixture
def client():
    """Run production route handlers and schemas against a known roster."""
    from api.dependencies import get_app_state
    from api.routes import analytics, compensation
    state = SimpleNamespace(
        has_data=lambda: True,
        analytics_engine=AnalyticsEngine(workforce()),
        compensation_engine=CompensationEngine(workforce()),
    )
    app = FastAPI()
    app.dependency_overrides[get_app_state] = lambda: state
    app.include_router(analytics.router)
    app.include_router(compensation.router)
    with TestClient(app) as instance:
        yield instance


def test_compensation_department_counts_reconcile_to_salary_population(client):
    summary = client.get('/api/compensation/summary').json()
    departments = client.get('/api/compensation/equity').json()
    # 6 known active employees, but only 4 measured positive salaries.
    assert summary['headcount'] == 4
    assert summary['total_payroll'] == 280000
    assert sum(row['headcount'] for row in departments) == 4
    assert {row['dept']: row['headcount'] for row in departments} == {'People': 2, 'Unknown': 2}
    assert sum(row['avg_salary'] * row['headcount'] for row in departments) == 280000


def test_compensation_tenure_counts_preserve_unmeasured_employees(client):
    response = client.get('/api/compensation/by-tenure')
    assert response.status_code == 200
    rows = response.json()
    assert sum(row['count'] for row in rows) == 4
    unknown = next(row for row in rows if row['tenure_bucket'] == 'Unknown')
    assert unknown['count'] == 2
    assert unknown['mean'] == 70000
    assert next(row for row in rows if row['tenure_bucket'] == '<1 year')['count'] == 1


def test_compensation_engine_reports_excluded_salary_denominator():
    result = CompensationEngine(workforce()).get_compensation_summary()
    assert result['active_count'] == 6
    assert result['salary_observations'] == 4
    assert result['excluded_salary_count'] == 2
    assert result['salary_coverage'] == pytest.approx(4 / 6)


def test_live_route_schemas_preserve_source_and_measurement_denominators(client):
    result = client.get('/api/compensation/summary').json()
    assert result['active_count'] == 6
    assert result['salary_observations'] == 4
    assert result['excluded_salary_count'] == 2
    assert result['salary_coverage'] == pytest.approx(4 / 6)
    result = client.get('/api/analytics/summary').json()
    assert result['record_count'] == 8
    assert result['attrition_known_count'] == 7
    assert result['observed_attrition_share'] == pytest.approx(1 / 7)
    assert result['salary_observations'] == 4
    assert result['salary_excluded_count'] == 2
    assert result['tenure_observations'] == 4
    assert result['tenure_excluded_count'] == 2


def test_department_median_reference_includes_unknown_department():
    ratios = CompensationEngine(workforce()).calculate_compa_ratio().set_index('EmployeeID')
    assert ratios.loc['0002', 'CompaRatio'] == pytest.approx(40000 / 70000)
    assert ratios.loc['0003', 'CompaRatio'] == pytest.approx(100000 / 70000)
    assert ratios['MetricSemantics'].eq('relative_to_department_median_not_formal_compa_ratio').all()


@pytest.mark.parametrize('tenure', [np.inf, -np.inf, 'unrecorded', None, -0.1])
def test_invalid_tenure_is_unknown_not_lost_or_assigned_to_long_service(tenure):
    frame = workforce()
    frame['Tenure'] = pd.Series([tenure] * len(frame), dtype=object)
    result = CompensationEngine(frame).get_salary_by_tenure()
    assert result.loc[result['TenureBucket'] == 'Unknown', 'Count'].iloc[0] == 4
    assert result['Count'].sum() == 4


def test_summary_measurement_counts_do_not_treat_missing_salary_as_zero():
    summary = AnalyticsEngine(workforce()).get_summary_statistics()
    assert summary['headcount'] == 6
    assert summary['salary_mean'] == 70000
    assert summary['salary_observations'] == 4
    assert summary['salary_excluded_count'] == 2


def test_invalid_rating_cannot_manufacture_an_eligible_group_comparison():
    frame = pd.DataFrame({
        'EmployeeID': [str(i) for i in range(14)], 'Attrition': 0,
        'Dept': ['A'] * 7 + ['B'] * 7,
        'LastRating': [1, 2, 3, 4, 5, 99, 100] + [1, 2, 3, 4, 5, -99, -100],
    })
    result = AnalyticsEngine(frame).compare_groups('Dept', 'LastRating')
    assert result['success'] is False  # Each group has only 5 valid ratings.


def test_valid_group_comparison_exposes_actual_measured_support():
    frame = pd.DataFrame({
        'EmployeeID': [str(i) for i in range(14)], 'Attrition': 0,
        'Dept': ['A'] * 7 + ['B'] * 7,
        'LastRating': [1, 2, 3, 4, 5, 2, 99] + [2, 3, 4, 5, 4, 3, -99],
    })
    result = AnalyticsEngine(frame).compare_groups('Dept', 'LastRating')
    assert result['success'] is True
    assert result['sample_size'] == 12
    assert result['group_observations'] == {'A': 6, 'B': 6}


def test_accuracy_cannot_hide_low_departure_recall():
    # Explicit confusion table: TP=8 FN=39 TN=243 FP=4, matching 294 examples.
    actual = np.array([1] * 47 + [0] * 247)
    probability = np.array([.8] * 8 + [.2] * 39 + [.2] * 243 + [.8] * 4)
    result = binary_metrics(actual, probability, training_prevalence=.16)
    assert result['accuracy'] == pytest.approx(251 / 294)
    assert result['recall'] == pytest.approx(8 / 47)
    assert result['precision'] == pytest.approx(8 / 12)
    assert result['confusion_matrix'] == {'true_positive': 8, 'false_negative': 39, 'true_negative': 243, 'false_positive': 4}
    assert result['majority_class_baseline_accuracy'] == pytest.approx(247 / 294)
    assert result['classification_threshold'] == .5
    assert result['threshold_selection'] == 'fixed_default_not_tuned_on_holdout'
    assert result['predicted_positive_count'] == 12
    assert result['recall_denominator'] == 47


def test_classification_boundary_is_inclusive_and_baseline_uses_training_data():
    result = binary_metrics([0, 0, 1], [.49, .5, .5], training_prevalence=.7)
    assert result['confusion_matrix'] == {'true_negative': 1, 'false_positive': 1, 'false_negative': 0, 'true_positive': 1}
    assert result['majority_class_baseline_accuracy'] == pytest.approx(1 / 3)


@pytest.mark.parametrize('actual,probability,prevalence', [
    ([], [], .2), ([0, .7], [.1, .6], .2), ([0, None], [.1, .6], .2),
    ([0, 1], [[.1], [.6]], .2), ([0, 1], [.1, .6], np.nan),
    ([0, 1], [.1, .6], 1.1), ([0, 1], [.1], .2),
])
def test_malformed_evaluation_data_cannot_generate_accuracy(actual, probability, prevalence):
    with pytest.raises(MLEngineError):
        binary_metrics(actual, probability, prevalence)


@pytest.fixture
def survey_runtime():
    from api.dependencies import get_app_state
    from api.routes import sentiment
    frame = pd.DataFrame({'EmployeeID': [f'{i:04d}' for i in range(20)], 'Dept': 'People', 'Attrition': 0})
    state = SimpleNamespace(raw_df=frame, has_data=lambda: True, enps_df=None, onboarding_df=None,
                            sentiment_engine=None)
    app = FastAPI()
    app.dependency_overrides[get_app_state] = lambda: state
    app.include_router(sentiment.router)
    with TestClient(app, raise_server_exceptions=False) as client:
        yield state, client


def survey_frame(kind):
    result = pd.DataFrame({'EmployeeID': [f'{i:04d}' for i in range(20)], 'SurveyDate': '2026-01-01'})
    if kind == 'enps':
        result['eNPSScore'] = 10
    else:
        result['SurveyType'] = '30-day'
        result['OverallScore'] = 4
    return result


@pytest.mark.parametrize('kind,missing', [
    ('enps', 'EmployeeID'), ('enps', 'SurveyDate'), ('enps', 'eNPSScore'),
    ('onboarding', 'EmployeeID'), ('onboarding', 'SurveyDate'),
    ('onboarding', 'SurveyType'), ('onboarding', 'OverallScore'),
])
def test_malformed_survey_upload_preserves_previous_valid_results(survey_runtime, kind, missing):
    state, client = survey_runtime
    frame = survey_frame(kind)
    upload_path = f'/api/sentiment/upload/{kind}'
    result_path = '/api/sentiment/enps' if kind == 'enps' else '/api/sentiment/onboarding/health'
    first = client.post(upload_path, files={'file': ('valid.csv', frame.to_csv(index=False).encode(), 'text/csv')})
    assert first.status_code == 200, first.text
    assert first.json()['success'] is True
    previous_engine = state.sentiment_engine
    previous_frame = getattr(state, f'{kind}_df')
    previous_results = client.get(result_path)
    assert previous_results.status_code == 200, previous_results.text
    assert previous_results.json()['available'] is True
    malformed = frame.drop(columns=missing)
    response = client.post(upload_path, files={'file': ('malformed.csv', malformed.to_csv(index=False).encode(), 'text/csv')})
    assert response.status_code == 400, response.text
    assert missing in response.json()['detail']
    assert state.sentiment_engine is previous_engine
    assert getattr(state, f'{kind}_df') is previous_frame
    assert client.get(result_path).json() == previous_results.json()


@pytest.mark.parametrize('kind', ['enps', 'onboarding'])
def test_survey_preparation_failure_cannot_replace_previous_valid_data(survey_runtime, monkeypatch, kind):
    state, client = survey_runtime
    frame = survey_frame(kind)
    upload_path = f'/api/sentiment/upload/{kind}'
    assert client.post(upload_path, files={'file': ('valid.csv', frame.to_csv(index=False).encode(), 'text/csv')}).status_code == 200
    previous_engine = state.sentiment_engine
    previous_frame = getattr(state, f'{kind}_df')
    def fail_preparation(*args, **kwargs):
        raise ValueError('Synthetic candidate preparation failure')
    monkeypatch.setattr('src.sentiment_engine.SentimentEngine', fail_preparation)
    frame['SurveyDate'] = '2026-02-01'
    response = client.post(upload_path, files={'file': ('next.csv', frame.to_csv(index=False).encode(), 'text/csv')})
    assert response.status_code == 500
    assert state.sentiment_engine is previous_engine
    assert getattr(state, f'{kind}_df') is previous_frame
