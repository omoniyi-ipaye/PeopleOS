"""Independent oracles for the second analytics integrity pass."""
import asyncio
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest


def staff(n=12):
    return pd.DataFrame({'EmployeeID': [f'E{i}' for i in range(n)], 'Dept': 'A',
                         'Tenure': 2., 'LastRating': 4., 'Salary': 100., 'Age': 30., 'Attrition': 0})


def test_team_percentages_use_measured_active_employee_denominator():
    from src.team_dynamics_engine import TeamDynamicsEngine
    df = staff(); df['LastRating'] = [5.] * 3 + [np.nan] * 8 + [1.]
    df.loc[11, 'Attrition'] = 1
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    row = TeamDynamicsEngine(df).identify_performance_variance().iloc[0]
    assert row['Headcount'] == 11
    assert row['RatingObservations'] == 3
    assert row['PercentHigh'] == 100


def test_team_empty_and_unmeasured_are_not_critical_or_healthy():
    from src.team_dynamics_engine import TeamDynamicsEngine
    empty = staff(0)
    assert TeamDynamicsEngine(empty).analyze_all()['health'].empty
    frame = staff().drop(columns=['Tenure', 'LastRating', 'Attrition'])
    row = TeamDynamicsEngine(frame).calculate_team_health_scores().iloc[0]
    assert pd.isna(row['HealthScore'])
    assert row['Status'] == 'Unavailable'


def test_succession_missing_ratings_are_unassessed_and_current_only():
    from src.succession_engine import SuccessionEngine
    frame = staff(); frame['LastRating'] = np.nan; frame.loc[0, 'Attrition'] = 1
    frame = pd.concat([frame, frame.iloc[[1]]], ignore_index=True)
    engine = SuccessionEngine(frame)
    assert len(engine.df) == 11
    assert set(engine.get_9box_matrix()['NineBox']) == {'Unassessed'}


def test_succession_summary_reads_actual_engine_keys_without_false_zero():
    from api.routes.succession import get_succession_summary
    engine = SimpleNamespace(df=staff(), analyze_all=lambda: {
        'readiness': pd.DataFrame({'ReadinessLevel': ['Ready Now', 'Developing']}),
        'gaps': pd.DataFrame({'Dept': ['A']}), 'bench_strength': pd.DataFrame(),
        'nine_box_summary': pd.DataFrame()})
    result = asyncio.run(get_succession_summary(state=SimpleNamespace(succession_engine=engine)))
    assert result['aggregate_ready_now_count'] == 1
    assert result['critical_gap_count'] == 1


def test_qoh_cohort_percentages_require_ratings_and_tenure():
    from src.quality_of_hire_engine import QualityOfHireEngine
    df = staff(60).assign(HireSource='Referral')
    df['LastRating'] = [5.] * 20 + [np.nan] * 40
    result = QualityOfHireEngine(df).analyze_cohort_performance().iloc[0]
    assert result['high_performer_pct'] == 100
    assert result['performance_observations'] == 20
    assert QualityOfHireEngine(df.drop(columns='Tenure')).analyze_cohort_performance().empty


def test_enps_grouping_cannot_multiply_survey_rows_by_employee_history():
    from src.sentiment_engine import SentimentEngine
    employees = pd.concat([staff(2), staff(2)], ignore_index=True)
    survey = pd.DataFrame({'EmployeeID': ['E0', 'E1'], 'eNPSScore': [10, 0], 'Dept': ['Survey A', 'Survey A']})
    result = SentimentEngine(employees, enps_df=survey).calculate_enps(group_by='Dept')
    assert result['total_responses'] == 2
    assert result['by_group'] == []
    assert result['suppressed_group_count'] == 1
    assert result['suppressed_response_count'] == 2


def test_missing_onboarding_dimensions_are_not_healthy():
    from src.sentiment_engine import SentimentEngine
    engine = SentimentEngine(staff(), onboarding_df=pd.DataFrame({'EmployeeID': ['E0'], 'SurveyType': ['30-day']}))
    result = engine.get_onboarding_health()
    assert result['overall_health'] == 'Unavailable'


def test_empty_vector_rebuild_clears_old_results_and_metadata():
    from src.vector_engine import VectorEngine
    engine = VectorEngine.__new__(VectorEngine)
    engine.index = object(); engine.metadata = [{'EmployeeID': 'old'}]
    engine.build_index([], [])
    assert not engine.is_initialized()
    assert engine.metadata == []


def test_vector_build_rejects_misaligned_metadata_without_model_access():
    from src.vector_engine import VectorEngine
    engine = VectorEngine.__new__(VectorEngine)
    engine.index = None; engine.metadata = []
    with pytest.raises(ValueError, match='align'):
        engine.build_index(['one', 'two'], [{'EmployeeID': 'E0'}])


def test_null_numeric_evidence_does_not_support_an_answer():
    from src.agent.aggregator import EvidenceAggregator
    from src.agent.evidence import ToolResult, EvidenceItem
    result = ToolResult(tool_id='test', status='success', summary='No measurement', evidence=[
        EvidenceItem(kind='derived', claim='Missing score', source_tool='test', metric='score', value=np.nan)])
    bundle = EvidenceAggregator().aggregate('What is the score?', [result])
    assert not bundle.can_synthesize()


def test_failed_and_wrong_dataset_evidence_cannot_support_synthesis():
    from src.agent.aggregator import EvidenceAggregator
    from src.agent.evidence import ToolResult, EvidenceItem
    failed = ToolResult(tool_id='failed', status='failed', summary='failed', evidence=[
        EvidenceItem(kind='observed', claim='Bad', source_tool='failed', metric='count', value=100)])
    wrong = ToolResult(tool_id='old', status='success', summary='old', evidence=[
        EvidenceItem(kind='observed', claim='Old', source_tool='old', metric='count', value=100, dataset_version='old')])
    empty = ToolResult(tool_id='empty', status='success', summary='empty')
    bundle = EvidenceAggregator().aggregate('Current?', [failed, wrong, empty], dataset_version='current')
    assert not bundle.can_synthesize()
    assert not bundle.evidence_items()


@pytest.fixture
def team_client():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from api.routes import team
    from src.team_dynamics_engine import TeamDynamicsEngine
    frame = staff(6)
    frame['LastRating'] = [5., 5., np.nan, np.nan, 99., 1.]
    frame['Gender'] = ['M', 'Female', None, 'Nonbinary', 'unknown', 'M']
    frame['Age'] = [24., 25., 40., 55., np.inf, 20.]
    frame['Pulse_Score'] = [5., 5., np.nan, 99., np.nan, 1.]
    frame.loc[5, 'Attrition'] = 1
    state = SimpleNamespace(raw_df=pd.concat([frame, frame.iloc[[0]]], ignore_index=True), team_dynamics_engine=TeamDynamicsEngine(frame))
    app = FastAPI(); app.include_router(team.router)
    app.dependency_overrides[team.require_data] = lambda: state
    app.dependency_overrides[team.require_team] = lambda: state
    with TestClient(app) as client:
        yield client


def test_comprehensive_team_api_reconciles_missing_and_measured_categories(team_client):
    response = team_client.get('/api/team/comprehensive')
    assert response.status_code == 200, response.text
    data = response.json()
    assert data['total_employees'] == 5
    assert data['rating_observations'] == 2
    assert data['top_performers_pct'] == 100
    assert data['gender_breakdown']['unknown'] == 2
    assert data['gender_breakdown']['other'] == 1
    assert sum(item['count'] for item in data['age_distribution']) == 5
    assert data['satisfaction']['avg_pulse'] == 5
    assert data['attrition_rate'] == pytest.approx(100 / 6, abs=.051)
    assert 'period_turnover' in data['attrition_rate_semantics']


def test_team_api_missing_filter_is_not_silently_ignored(team_client):
    response = team_client.get('/api/team/comprehensive?countries=Spain')
    assert response.status_code == 400
    assert 'missing Country' in response.json()['detail']


def test_team_api_invalid_or_empty_measurements_serialize_as_null(team_client):
    response = team_client.get('/api/team/health')
    assert response.status_code == 200, response.text
    assert response.json()[0]['headcount'] == 5
    response = team_client.get('/api/team/comprehensive?departments=Absent')
    assert response.status_code == 200, response.text
    assert response.json()['total_employees'] == 0
    assert response.json()['avg_rating'] is None


def test_readiness_depends_on_recorded_assessment_not_risk_or_tenure():
    from src.succession_engine import SuccessionEngine
    raw = staff(4)
    assert SuccessionEngine(raw).calculate_readiness_scores()['ReadinessScore'].isna().all()
    raw['SuccessionReadiness'] = ['Ready Now', 'Developing', None, 'invalid']
    row = SuccessionEngine(raw).calculate_bench_strength().iloc[0]
    assert row['Total'] == 4 and row['Assessed'] == 2 and row['Unassessed'] == 2
    assert row['AssessmentCoverage'] == .5
    assert row['BenchStrength'] == .65  # (1 + .3) / 2 assessed records
    raw['Tenure'] = 10000
    raw['LastRating'] = 1
    assert SuccessionEngine(raw).calculate_bench_strength().iloc[0]['BenchStrength'] == .65


def test_onboarding_uses_latest_dated_stage_and_ignores_unknown_stages():
    from src.sentiment_engine import SentimentEngine
    survey = pd.DataFrame({'EmployeeID': ['E0'] * 4, 'SurveyType': ['30-day', '30-day', '60-day', 'other'],
                           'OverallScore': [5., 1., 2., 5.],
                           'SurveyDate': ['2024-02-02', '2024-02-01', '2024-03-01', '2024-04-01']})
    result = SentimentEngine(staff(), onboarding_df=survey).analyze_onboarding_trajectory()['trajectories'][0]
    assert result['scores'] == {'30-day': 5., '60-day': 2.}
    assert result['trend_change'] == -3
    assert result['surveys_completed'] == 2


@pytest.mark.parametrize('missing', [np.inf, -np.inf, pd.NA, pd.NaT])
def test_nonfinite_and_nullable_evidence_never_becomes_support(missing):
    from src.agent.evidence import EvidenceItem, ToolResult
    from src.agent.aggregator import EvidenceAggregator
    item = EvidenceItem(kind='observed', source_tool='t', metric='score', claim='Unmeasured', value=missing)
    assert item.value is None
    bundle = EvidenceAggregator().aggregate('Score?', [ToolResult(tool_id='t',status='success',summary='missing',evidence=[item])])
    assert not bundle.can_synthesize()


def test_vector_geometry_metadata_copy_and_failed_rebuild():
    from src.vector_engine import VectorEngine
    class Model:
        def encode(self, texts, **kwargs):
            return np.array([{'a':[0.,0.], 'b':[3.,4.], 'query':[0.,0.], 'bad':[np.nan,0.]}[x] for x in texts])
    class Index:
        def __init__(self, dim): pass
        def add(self, values): self.values = values
        def search(self, query, k):
            d = ((self.values - query[0]) ** 2).sum(axis=1)
            ids = np.argsort(d)[:k]
            return d[ids][None, :], ids[None, :]
    engine = VectorEngine.__new__(VectorEngine)
    engine.model = Model(); engine._faiss = SimpleNamespace(IndexFlatL2=Index)
    metadata = [{'EmployeeID':'E0'}, {'EmployeeID':'E1'}]
    engine.build_index(['a','b'], metadata)
    metadata[0]['EmployeeID'] = 'mutated'
    result = engine.search('query', top_k=20)
    assert result[0]['EmployeeID'] == 'E0'
    assert result[1]['squared_l2_distance'] == 25
    assert result[1]['similarity_score'] == pytest.approx(1 / 26)
    assert 'not_probability' in result[1]['score_semantics']
    with pytest.raises(ValueError): engine.build_index(['bad'], [{'EmployeeID':'E2'}])
    assert not engine.is_initialized() and not engine.metadata


def test_succession_api_preserves_unassessed_and_separate_potential():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from api.routes import succession
    from src.succession_engine import SuccessionEngine
    frame = staff(4); frame['PotentialRating'] = [1., 5., None, 99.]
    state = SimpleNamespace(succession_engine=SuccessionEngine(frame))
    app = FastAPI(); app.include_router(succession.router)
    app.dependency_overrides[succession.require_succession] = lambda: state
    with TestClient(app) as client:
        response = client.get('/api/succession/bench-strength')
        assert response.status_code == 200, response.text
        row = response.json()[0]
        assert row['bench_strength'] is None and row['unassessed'] == 4
        assert client.get('/api/succession/gaps').json() == []
        matrix = {row['category']: row for row in client.get('/api/succession/9box/summary').json()}
        assert matrix['Stars']['count'] == 1
        assert matrix['Solid Performers']['count'] == 1
        assert matrix['Unassessed']['count'] == 2
        assert sum(row['percentage'] for row in matrix.values()) == 100
        state.succession_engine = SuccessionEngine(frame.iloc[:0])
        response = client.get('/api/succession/summary')
        assert response.status_code == 200, response.text
        assert response.json()['total_employees'] == 0


def test_search_failure_is_unavailable_not_a_successful_empty_result():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from api.routes import search
    def fail(*args, **kwargs): raise RuntimeError('embedding failed')
    app = FastAPI(); app.include_router(search.router)
    app.dependency_overrides[search.require_vector_search] = lambda: SimpleNamespace(vector_engine=SimpleNamespace(search=fail))
    with TestClient(app) as client:
        response = client.post('/api/search?query=manager')
        assert response.status_code == 503
        assert 'unavailable' in response.json()['detail']


def test_enps_date_range_uses_whole_utc_day_and_excludes_invalid_trend_dates():
    from src.sentiment_engine import SentimentEngine
    frame = pd.DataFrame({'EmployeeID': ['E0','E1','E2'], 'eNPSScore': [10,0,0],
                          'SurveyDate': ['2024-01-31T23:30:00Z','2024-02-01T01:00:00Z','invalid']})
    engine = SentimentEngine(staff(), enps_df=frame)
    result = engine.calculate_enps(date_from='2024-01-31', date_to='2024-01-31')
    assert result['total_responses'] == 1 and result['overall_enps'] == 100
    trends = engine.get_enps_trends()
    assert [row['period'] for row in trends['trends']] == ['2024-01', '2024-02']
    assert trends['recent_change'] == -200
    undated = SentimentEngine(staff(), enps_df=frame.drop(columns='SurveyDate'))
    assert not undated.calculate_enps(date_from='2024-01-31')['available']


def test_agent_keeps_current_evidence_once_and_rejects_wrong_model_or_source():
    from src.agent.evidence import EvidenceItem, ToolResult
    from src.agent.aggregator import EvidenceAggregator
    def item(source='t', model='current'):
        return EvidenceItem(kind='observed', source_tool=source, model_version=model,
                            dataset_version='data', metric='count', claim='Observed count', value=10)
    good = ToolResult(tool_id='t', status='success', summary='Current count', evidence=[item()])
    wrong_source = ToolResult(tool_id='t', status='success', summary='Wrong tool', evidence=[item(source='other')])
    wrong_model = ToolResult(tool_id='t', status='success', summary='Old model', evidence=[item(model='old')])
    bundle = EvidenceAggregator().aggregate('Count?', [good, good, wrong_source, wrong_model],
                                             dataset_version='data', model_version='current')
    assert len(bundle.tool_results) == 3
    assert [e.evidence_id for e in bundle.evidence_items()] == [good.evidence[0].evidence_id]
    assert len(bundle.unknowns) == 2
