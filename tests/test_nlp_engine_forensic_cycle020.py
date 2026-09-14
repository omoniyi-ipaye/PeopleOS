"""Cycle 020 forensic contracts for bounded, exploratory NLP output.

These tests verify response integrity and evidence boundaries. They do not
claim that an LLM is accurate on workforce text, multilingual input, irony,
negation, or any production population.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes import nlp as routes
from src.nlp_engine import NLPEngine, NLPEngineError
from src.platform.provenance import frame_fingerprint


def reviews(count: int = 2) -> pd.DataFrame:
    return pd.DataFrame({
        'EmployeeID': [f'E{i}' for i in range(count)],
        'PerformanceText': ['Python delivery was strong.' for _ in range(count)],
        'Dept': ['People' for _ in range(count)],
    })


class FixtureClient:
    is_available = True
    model = 'cycle-020-fixture'

    def __init__(self, responses):
        self.responses = list(responses)
        self.client = self

    def generate(self, **kwargs):
        response = self.responses.pop(0)
        return {'response': json.dumps(response)}


def engine_for(*responses) -> NLPEngine:
    return NLPEngine(FixtureClient(responses))


def test_invalid_nlp_configuration_fails_closed_before_model_calls(monkeypatch):
    monkeypatch.setattr('src.nlp_engine.load_config', lambda: {'nlp': {'batch_size': 0}})
    with pytest.raises(NLPEngineError, match='batch_size'):
        NLPEngine(None)


def test_invalid_review_identity_does_not_reach_the_model():
    engine = engine_for([])
    source = reviews().assign(EmployeeID=['E0', 'E0'])
    assert engine.analyze_sentiment(source).empty
    assert engine._last_sentiment_input_count == 0


def test_sentiment_batch_requires_exact_identity_coverage():
    engine = engine_for([{
        'EmployeeID': 'E0', 'sentiment_score': 0.9, 'sentiment_label': 'Positive',
    }])
    with pytest.raises(NLPEngineError, match='cover every requested review'):
        engine._analyze_sentiment_batch(['Python', 'Python'], ['E0', 'E1'])


@pytest.mark.parametrize('score', [True, float('nan'), float('inf'), -0.1, 1.1])
def test_sentiment_rejects_boolean_nonfinite_and_out_of_range_scores(score):
    engine = engine_for([{
        'EmployeeID': 'E0', 'sentiment_score': score, 'sentiment_label': 'Positive',
    }])
    with pytest.raises(NLPEngineError, match='invalid identity or score'):
        engine._analyze_sentiment_batch(['Python'], ['E0'])


def test_failed_sentiment_batch_is_missing_and_reported_as_unprocessed():
    engine = engine_for([{
        'EmployeeID': 'E0', 'sentiment_score': 0.9, 'sentiment_label': 'Positive',
    }])
    result = engine.analyze_sentiment(reviews())
    assert result.empty
    summary = engine.get_sentiment_summary(result)
    assert summary['sentiment_observations'] == 0
    assert summary['unprocessed_sentiment_rows'] == 2
    assert summary['avg_sentiment'] is None


def test_sentiment_success_preserves_exact_ids_and_observation_counts():
    engine = engine_for([
        {'EmployeeID': 'E0', 'sentiment_score': 0.9, 'sentiment_label': 'Positive'},
        {'EmployeeID': 'E1', 'sentiment_score': 0.2, 'sentiment_label': 'Negative'},
    ])
    result = engine.analyze_sentiment(reviews())
    assert result.EmployeeID.tolist() == ['E0', 'E1']
    assert engine.get_sentiment_summary(result)['sentiment_observations'] == 2
    assert engine.get_sentiment_summary(result)['unprocessed_sentiment_rows'] == 0


def test_large_sentiment_population_is_bounded_and_disclosed():
    engine = engine_for([
        {'EmployeeID': f'E{i}', 'sentiment_score': 0.9, 'sentiment_label': 'Positive'}
        for i in range(10)
    ])
    result = engine.analyze_sentiment(reviews(800))
    summary = engine.get_sentiment_summary(result)

    assert len(result) == 10
    assert engine._last_sentiment_input_count == 800
    assert engine._last_sentiment_excluded_count == 790
    assert summary['unprocessed_sentiment_rows'] == 790


def test_skill_output_is_bounded_and_must_be_literal_source_evidence():
    source = reviews(1)
    hallucinated = engine_for({'technical_skills': ['Python', 'Java'], 'soft_skills': []})
    with pytest.raises(NLPEngineError, match='not literally supported'):
        hallucinated.extract_skills(source)

    skills = [f'Skill{i}' for i in range(16)]
    over_limit = engine_for({'technical_skills': skills, 'soft_skills': []})
    with pytest.raises(NLPEngineError, match='15-skill category limit'):
        over_limit.extract_skills(source.assign(PerformanceText=' '.join(skills)))


def test_skill_counts_are_source_review_presence_counts_not_token_frequency():
    engine = engine_for({'technical_skills': ['Python'], 'soft_skills': []})
    result = engine.extract_skills(pd.DataFrame({
        'EmployeeID': ['E0', 'E1'],
        'PerformanceText': ['Python Python', 'No matching skill'],
    }))
    assert result['skill_counts'] == {'Python': 1}
    assert result['skill_count_semantics'] == 'source_review_presence_count'
    assert result['review_observations'] == 2


def test_topics_ignore_model_prevalence_and_disclose_sampling_scope():
    engine = engine_for([{
        'name': 'Support', 'description': 'Team support', 'prevalence': '99%', 'sentiment': 'Mixed',
    }])
    topic = engine.extract_topics(reviews(60))[0]
    assert topic['prevalence'] is None
    assert topic['measurement_semantics'] == 'generated_theme_not_measured_prevalence'
    assert topic['sample_size'] == 10
    assert topic['sample_scope'] == 'first_10_nonempty_unique_employee_reviews'
    assert routes.TopicInfo(**topic).prevalence is None


def test_duplicate_topic_names_fail_closed():
    engine = engine_for([
        {'name': 'Support', 'description': 'One', 'sentiment': 'Neutral'},
        {'name': ' support ', 'description': 'Two', 'sentiment': 'Neutral'},
    ])
    with pytest.raises(NLPEngineError, match='duplicate theme names'):
        engine.extract_topics(reviews())


def test_process_all_reports_component_boundaries_without_promoting_partial_results():
    engine = engine_for(
        [
            {'EmployeeID': 'E0', 'sentiment_score': 0.9, 'sentiment_label': 'Positive'},
            {'EmployeeID': 'E1', 'sentiment_score': 0.2, 'sentiment_label': 'Negative'},
        ],
        {'technical_skills': ['Python'], 'soft_skills': []},
        [{'name': 'Delivery', 'description': 'Delivery work', 'sentiment': 'Neutral'}],
    )
    result = engine.process_all(reviews())
    assert result['analysis_status'] == 'available'
    assert {key: value['status'] for key, value in result['component_status'].items()} == {
        'sentiment': 'available', 'skills': 'available', 'topics': 'available',
    }

    partial_engine = engine_for(
        [{'EmployeeID': 'E0', 'sentiment_score': 0.9, 'sentiment_label': 'Positive'}],
        {'technical_skills': ['Java'], 'soft_skills': []},
        [],
        [{'name': 'Delivery', 'description': 'Delivery work', 'sentiment': 'Neutral'}],
    )
    partial_engine.batch_size = 1
    partial = partial_engine.process_all(reviews())
    assert partial['analysis_status'] == 'partial'
    assert partial['sentiment_summary']['unprocessed_sentiment_rows'] == 1
    assert partial['skills']['status'] == 'unavailable'


def state_with_cache(cache):
    source = reviews()
    return SimpleNamespace(
        raw_df=source,
        nlp_results=cache,
        nlp_engine=SimpleNamespace(process_all=lambda source: pytest.fail('cache should be rejected before inference')),
        has_data=lambda: True,
        load_from_database=lambda: False,
        runtime_provenance={
            'workspace_id': 'local',
            'dataset_id': 'ds-cycle-020',
            'generation': 'cycle-020',
            'current_fingerprint': frame_fingerprint(source),
        },
    )


def test_same_snapshot_malformed_cache_is_rejected_and_not_reused():
    source = reviews()
    state = state_with_cache({
        'provenance': {
            'workspace_id': 'local', 'dataset_id': 'ds-cycle-020',
            'generation': 'cycle-020', 'current_fingerprint': frame_fingerprint(source),
        },
        'sentiment_summary': {'positive_count': 0},
        'topics': [{'name': 'Broken'}],
        'skills': {},
        'nlp_available': True,
    })
    app = FastAPI()
    app.dependency_overrides[routes.get_app_state] = lambda: state
    app.include_router(routes.router)
    with TestClient(app) as client:
        response = client.get('/api/nlp/analysis')
    assert response.status_code == 409
    assert 'integrity validation' in response.json()['detail']
    assert state.nlp_results is None


def test_nlp_api_exposes_provenance_and_component_status_without_prevalence_claims():
    source = reviews()
    state = state_with_cache(None)
    state.nlp_engine = SimpleNamespace(process_all=lambda frame: {
        'sentiment_summary': {
            'avg_sentiment': None, 'positive_count': 0, 'neutral_count': 0,
            'negative_count': 0, 'positive_pct': 0, 'neutral_pct': 0, 'negative_pct': 0,
            'sentiment_observations': 0, 'unprocessed_sentiment_rows': 2,
        },
        'topics': [{
            'name': 'Support', 'description': 'Team support', 'prevalence': None,
            'measurement_semantics': 'generated_theme_not_measured_prevalence',
            'sample_size': 2, 'sample_scope': 'first_10_nonempty_unique_employee_reviews',
        }],
        'skills': {'technical_skills': [], 'soft_skills': [], 'skill_counts': {}},
        'nlp_available': True, 'analysis_status': 'partial',
        'component_status': {'sentiment': {'status': 'unavailable'}},
    })
    app = FastAPI()
    app.dependency_overrides[routes.get_app_state] = lambda: state
    app.include_router(routes.router)
    with TestClient(app) as client:
        response = client.get('/api/nlp/analysis')
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload['provenance']['dataset_id'] == 'ds-cycle-020'
    assert payload['analysis_status'] == 'partial'
    assert payload['sentiment_summary']['unprocessed_sentiment_rows'] == 2
    assert payload['topics'][0]['prevalence'] is None
    assert payload['topics'][0]['sample_scope'] == 'first_10_nonempty_unique_employee_reviews'
    assert state.nlp_results['provenance'] == payload['provenance']
