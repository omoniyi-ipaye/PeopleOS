"""Cycle 023 forensic contracts for SentimentEngine and its governed API routes."""

from types import SimpleNamespace

import pandas as pd
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.sentiment_engine import SentimentEngine


def workforce(size=15):
    return pd.DataFrame({
        'EmployeeID': [f'E{i}' for i in range(size)],
        'Dept': 'People',
        'Attrition': 0,
    })


def test_onboarding_latest_response_is_the_only_snapshot_measure_and_is_audited():
    survey = pd.DataFrame({
        'EmployeeID': ['E0', 'E0', 'E1', 'E1'],
        'SurveyType': ['30-day', '30-day', '30-day', '30-day'],
        'SurveyDate': ['2026-01-01', '2026-02-01', '2026-01-01', '2026-01-01'],
        'OverallScore': [1.0, 5.0, 3.0, 4.0],
        'ManagerSupport': [1.0, 5.0, 3.0, 4.0],
    })

    engine = SentimentEngine(workforce(), onboarding_df=survey)
    health = engine.get_onboarding_health()
    trajectory = engine.analyze_onboarding_trajectory()['trajectories']

    assert health['by_survey_type'] == [{
        'survey_type': '30-day',
        'avg_score': 4.5,
        'responses': 2,
        'healthy_pct': 100.0,
    }]
    assert health['dimension_scores'][0]['avg_score'] == 4.5
    assert health['response_coverage']['onboarding']['by_survey_type']['30-day']['unique_respondents'] == 2
    assert health['response_coverage']['onboarding']['by_survey_type']['30-day']['response_rate_pct'] == 13.3
    assert health['response_coverage']['onboarding']['dimensions']['ManagerSupport']['observations'] == 2
    assert health['survey_coverage']['onboarding']['superseded_rows'] == 2
    assert health['survey_coverage']['onboarding']['latest_response_rows'] == 2
    assert health['response_coverage']['onboarding']['response_weighting'].startswith('latest_response')
    assert {row['latest_score'] for row in trajectory} == {5.0, 4.0}


def test_onboarding_date_ties_and_missing_dates_are_visible_and_deterministic():
    survey = pd.DataFrame({
        'EmployeeID': ['E0', 'E0', 'E1'],
        'SurveyType': ['30-day', '30-day', '30-day'],
        'SurveyDate': ['2026-01-01', '2026-01-01', None],
        'OverallScore': [2.0, 4.0, 3.0],
    })

    engine = SentimentEngine(workforce(), onboarding_df=survey)
    selected = engine.analyze_onboarding_trajectory()['trajectories']
    selected_by_id = {row['EmployeeID']: row['latest_score'] for row in selected}
    coverage = engine.response_coverage['onboarding']

    assert selected_by_id == {'E0': 4.0, 'E1': 3.0}
    assert coverage['date_tie_rows'] == 1
    assert coverage['undated_rows'] == 1
    assert 'same-date ties retain the last source row' in coverage['response_selection']


def test_onboarding_totals_are_computed_before_display_truncation():
    survey = pd.DataFrame({
        'EmployeeID': [f'E{i}' for i in range(15)],
        'SurveyType': '30-day',
        'SurveyDate': '2026-01-01',
        'OverallScore': 2.0,
    })

    engine = SentimentEngine(workforce(), onboarding_df=survey)
    result = engine.analyze_onboarding_trajectory()
    warnings = engine.detect_early_warnings()

    assert result['summary']['at_risk_count'] == 15
    assert result['at_risk_employee_count'] == 15
    assert result['at_risk_employees_returned'] == 10
    assert result['at_risk_employees_truncated'] is True
    assert len(result['at_risk_employees']) == 10
    assert warnings['summary']['total_observed_flags'] == 15
    assert warnings['summary']['total_at_risk'] == len(warnings['warnings']) == 15


def test_low_score_and_decline_warning_labels_do_not_claim_future_departure():
    survey = pd.DataFrame({
        'EmployeeID': ['E0', 'E0', 'E1', 'E1'],
        'SurveyType': ['30-day', '60-day', '30-day', '60-day'],
        'SurveyDate': ['2026-01-01', '2026-02-01', '2026-01-01', '2026-02-01'],
        'OverallScore': [4.0, 2.0, 2.0, 2.0],
    })

    result = SentimentEngine(workforce(), onboarding_df=survey).detect_early_warnings()
    by_id = {warning['EmployeeID']: warning for warning in result['warnings']}

    assert by_id['E0']['warning_type'] == 'Declining Onboarding'
    assert by_id['E1']['warning_type'] == 'Low Onboarding Score'
    assert all('not a prediction of future departure' in warning['details'] for warning in result['warnings'])
    assert result['claim_boundary'].startswith('Observed survey flags')
    assert result['summary']['warning_types'] == ['Declining Onboarding', 'Low Onboarding Score']


def test_enps_reports_repeated_response_weighting_and_nonresponse_boundary():
    survey = pd.DataFrame({
        'EmployeeID': ['E0', 'E0', 'E1'],
        'SurveyDate': ['2026-01-01', '2026-02-01', '2026-02-01'],
        'eNPSScore': [10, 0, 10],
    })

    engine = SentimentEngine(workforce(), enps_df=survey)
    result = engine.calculate_enps()
    coverage = result['analysis_coverage']

    assert result['total_responses'] == 3
    assert result['overall_enps'] == 33.3
    assert coverage['unique_respondents'] == 2
    assert coverage['repeated_response_rows'] == 1
    assert coverage['response_rate_pct'] == 13.3
    assert result['response_coverage']['enps']['response_weighting'].startswith('each_valid_response_weighted_equally')
    assert 'no nonresponse-bias adjustment' in result['nonresponse_boundary']

    filtered = engine.calculate_enps(date_from='2026-02-01', date_to='2026-02-01')
    assert filtered['analysis_coverage']['analysis_rows'] == 2
    assert filtered['analysis_coverage']['unique_respondents'] == 2
    assert filtered['analysis_coverage']['repeated_response_rows'] == 0
    trends = engine.get_enps_trends()
    february = next(point for point in trends['trends'] if point['period'] == '2026-02')
    assert february['responses'] == 2
    assert february['unique_respondents'] == 2


def _sentiment_client(engine, raw, *, with_provenance=True):
    from api.routes import sentiment

    runtime = {
        'workspace_id': 'local',
        'dataset_id': 'ds_cycle023',
        'dataset_version': 23,
        'generation': 'generation-cycle023',
        'current_fingerprint': engine.population_fingerprint,
        'source_name': 'cycle023.csv',
    } if with_provenance else None
    state = SimpleNamespace(
        raw_df=raw,
        has_data=lambda: True,
        load_from_database=lambda: True,
        sentiment_engine=engine,
        runtime_provenance=runtime,
        enps_df=None,
        onboarding_df=None,
    )
    app = FastAPI()
    app.dependency_overrides[sentiment.get_app_state] = lambda: state
    app.include_router(sentiment.router)
    return state, TestClient(app, raise_server_exceptions=False)


def test_sentiment_route_exposes_verified_dataset_and_no_cache_provenance():
    raw = workforce()
    engine = SentimentEngine(raw, onboarding_df=pd.DataFrame({
        'EmployeeID': ['E0'],
        'SurveyType': ['30-day'],
        'SurveyDate': ['2026-01-01'],
        'OverallScore': [4.0],
    }))
    _, client = _sentiment_client(engine, raw)

    response = client.get('/api/sentiment/onboarding/health')
    assert response.status_code == 200, response.text
    body = response.json()
    assert body['data_provenance']['status'] == 'verified'
    assert body['data_provenance']['dataset_id'] == 'ds_cycle023'
    assert body['data_provenance']['cache']['used'] is False
    assert body['by_survey_type'][0]['response_rate_pct'] == 6.7


def test_sentiment_route_rejects_engine_after_active_snapshot_changes():
    raw = workforce()
    engine = SentimentEngine(raw, onboarding_df=pd.DataFrame({
        'EmployeeID': ['E0'],
        'SurveyType': ['30-day'],
        'SurveyDate': ['2026-01-01'],
        'OverallScore': [4.0],
    }))
    state, client = _sentiment_client(engine, raw)
    state.raw_df.loc[0, 'Dept'] = 'Changed after activation'

    response = client.get('/api/sentiment/onboarding/health')
    assert response.status_code == 409
    assert 'changed after activation' in response.json()['detail']
