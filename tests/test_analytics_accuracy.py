"""Independent, hand-calculated oracles for analytics correctness."""

import asyncio
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.analytics_engine import AnalyticsEngine
from src.compensation_engine import CompensationEngine
from src.experience_engine import ExperienceEngine
from src.fairness_engine import FairnessEngine
from src.survival_engine import SurvivalEngine
from src.quality_of_hire_engine import QualityOfHireEngine
from src.sentiment_engine import SentimentEngine
from src.forecasting_engine import ForecastingEngine
from src.scenario_engine import ScenarioEngine


def rows(n=40):
    return pd.DataFrame({'EmployeeID': [f'E{i}' for i in range(n)], 'Dept': 'A',
                         'Salary': 100., 'Age': 30, 'LastRating': 3.,
                         'Tenure': 2., 'Attrition': 0})


def test_active_salary_statistics_agree_and_exclude_nonfinite_values():
    frame = rows(6)
    frame['Salary'] = [100, 200, -50, 0, np.inf, 9999]
    frame.loc[5, 'Attrition'] = 1
    analytics = AnalyticsEngine(frame)
    compensation = CompensationEngine(frame)
    assert analytics.get_summary_statistics()['salary_mean'] == 150
    assert analytics.get_department_aggregates().iloc[0]['Avg_Salary'] == 150
    assert compensation.get_compensation_summary()['total_payroll'] == 300


def test_kaplan_meier_integrates_steps_and_reports_supported_horizons():
    frame = rows(60)
    frame['Tenure'] = [1.] * 30 + [2.] * 30
    frame['Attrition'] = [1] * 30 + [0] * 30
    result = SurvivalEngine(frame).fit_kaplan_meier()['overall']
    # S(t)=1 on [0,12), then .5 on [12,24]; area = 12 + 6.
    assert result['mean_survival_months'] == pytest.approx(18)
    assert result['survival_at_12mo'] == .5
    assert result['survival_at_60mo'] is None


def test_survival_interval_arrays_match_the_entire_curve():
    frame = rows(60)
    frame['Tenure'] = np.arange(1, 61) / 12
    frame['Attrition'] = [0, 1] * 30
    result = SurvivalEngine(frame).fit_kaplan_meier()['overall']
    assert len(result['confidence_intervals']['lower']) == len(result['survival_function'])


def test_experience_nonrespondents_are_not_assigned_proxy_scores():
    frame = rows()
    frame['Pulse_Score'] = [5.] * 10 + [np.nan] * 30
    engine = ExperienceEngine(frame)
    result = engine.calculate_experience_index()
    assert result['overall_exi'] == 100
    assert result['respondent_count'] == 10
    assert result['response_coverage'] == .25
    assert engine.df['_exi_score'].notna().sum() == 10


def test_experience_invalid_signals_remain_unavailable():
    frame = rows()
    frame['Pulse_Score'] = [np.nan, 9, -1, np.inf] * 10
    assert ExperienceEngine(frame).calculate_experience_index()['available'] is False


def test_experience_fractional_band_boundaries_count_every_respondent():
    frame = rows(10)
    frame['Pulse_Score'] = [1.79, 2.59, 3.39, 4.19, 5] * 2
    result = ExperienceEngine(frame).get_engagement_segments()
    assert sum(s['count'] for s in result['segments']) == 10


def test_undefined_fairness_ratio_is_not_a_clean_bill_of_health():
    frame = rows()
    frame['Gender'] = ['M'] * 20 + ['F'] * 20
    frame['Attrition'] = 1
    engine = FairnessEngine(frame)
    assert engine.get_fairness_summary()['overall_status'] == 'Insufficient evidence'


def test_fairness_age_boundaries_match_labels():
    frame = rows(5)
    frame['Age'] = [29, 30, 40, 50, 60]
    assert FairnessEngine(frame).df['Age_Group'].astype(str).tolist() == ['Under 30', '30-39', '40-49', '50-59', '60+']


def test_duplicate_predictions_cannot_inflate_fairness_sample_size():
    frame = rows()
    frame['Gender'] = 'F'
    predictions = pd.DataFrame({'EmployeeID': ['E0'] * 20, 'risk_score': .9})
    assert FairnessEngine(frame, predictions).analyze_prediction_fairness()['available'] is False


def test_hiring_source_quality_ignores_missing_components():
    frame = rows()
    frame['HireSource'] = 'Referral'
    frame['LastRating'] = np.nan
    # Only measured component is retained share = 100%.
    result = QualityOfHireEngine(frame).calculate_source_effectiveness()
    assert result.iloc[0]['quality_score'] == 100


def test_constant_group_comparison_is_unavailable_not_nonsignificant():
    frame = rows()
    frame['Dept'] = ['A', 'B'] * 20
    result = AnalyticsEngine(frame).compare_groups('Dept', 'Salary')
    assert result['success'] is False


def test_forecast_uses_active_monthly_censuses_and_does_not_invent_missing_months():
    history = pd.concat([rows(20).assign(SnapshotDate=f'2024-{month:02d}-28', Attrition=[0] * 10 + [1] * 10)
                         for month in range(1, 13)], ignore_index=True)
    result = ForecastingEngine(history).forecast_metric('headcount', periods=3)
    assert result['success'] is True
    assert all(p['value'] == 10 for p in result['history'])
    assert len(result['forecast']) == 3
    assert result['model'] == 'last_observation_baseline'
    assert all('lower' not in p for p in result['forecast'])
    assert not ForecastingEngine(history).forecast_metric('turnover_rate')['success']
    gap = history[history['SnapshotDate'] != '2024-06-28']
    assert not ForecastingEngine(gap).forecast_metric('headcount')['success']


@pytest.mark.parametrize('months,expected', [(6, 6000), (12, 12000), (24, 24000)])
def test_scenario_annual_salary_and_horizon_costs(months, expected):
    frame = rows(12)
    frame['Salary'] = 10000.
    result = ScenarioEngine(frame).simulate_compensation_change(
        adjustment_type='percentage', target={'scope': 'all'}, adjustment_value=10., time_horizon_months=months)
    assert result.cost_impact.salary_change == expected
    assert result.cost_impact.total_cost == expected
    assert result.turnover_change == 0  # No assumed efficacy supplied.


def test_monte_carlo_uses_supplied_cohort_cost_and_does_not_reset_global_rng():
    engine = ScenarioEngine(rows())
    np.random.seed(9)
    expected = np.random.random()
    np.random.seed(9)
    simulation = engine._run_monte_carlo(.1, 0, 1000, 10, 50, baseline_outcome=.2)
    assert simulation.cost_impact_mean == pytest.approx(950)
    assert np.random.random() == expected


def test_enps_invalid_responses_do_not_become_passives():
    survey = pd.DataFrame({'EmployeeID': [f'E{i}' for i in range(50)],
                           'eNPSScore': [10] * 20 + [0] * 10 + [np.nan, 99] * 10})
    engine = SentimentEngine(employee_df=survey[['EmployeeID']], enps_df=survey)
    assert len(engine.enps_df) == 30
    result = engine.calculate_enps()
    assert result['overall_enps'] == pytest.approx(33.3, abs=.1)


def test_constant_pay_groups_have_descriptive_gap_without_invalid_inference():
    frame = rows()
    frame['Gender'] = ['M', 'F'] * 20
    result = CompensationEngine(frame).calculate_gender_pay_gap()
    assert result['raw_gap_pct'] == 0
    assert result['p_value'] is None
    assert result['inference_available'] is False
    import json
    json.dumps(result, allow_nan=False)


def test_experience_coverage_counts_valid_onboarding_responses():
    frame = rows()
    frame['Onboarding_30d'] = [5.] * 10 + [99.] * 30
    engine = ExperienceEngine(frame)
    assert engine.get_available_signals()['coverage_percentage'] == 25


def test_headcount_financial_summary_and_simulation_use_same_assumptions():
    result = ScenarioEngine(rows()).simulate_headcount_change('expansion', {'scope': 'all'}, change_count=5)
    assert result.simulation.cost_impact_mean == pytest.approx(result.cost_impact.net_impact)
    assert result.simulation.roi_mean == pytest.approx(result.roi_estimate, abs=.051)


def test_promotion_comparison_does_not_claim_unperformed_controls():
    from src.structural_engine import StructuralEngine
    frame = rows(60)
    frame['Gender'] = ['M', 'F'] * 30
    frame['YearsSinceLastPromotion'] = np.arange(60) / 10
    result = StructuralEngine(frame).audit_promotion_velocity()
    assert result['available']
    assert result['audit_results'][0]['controls_applied'] is False
    assert 'Unadjusted' in result['methodology']


def test_network_api_does_not_infer_collaboration_from_departments():
    from api.routes.network import get_network_summary, get_key_influencers
    from fastapi import HTTPException
    assert asyncio.run(get_network_summary(state=None))['available'] is False
    with pytest.raises(HTTPException) as exc:
        asyncio.run(get_key_influencers(state=None))
    assert exc.value.status_code == 409


def test_failed_and_hallucinated_nlp_are_missing_not_neutral():
    from src.nlp_engine import NLPEngine
    class FakeClient:
        is_available = True
        model = 'synthetic-test'
        def __init__(self):
            self.client = self
        def generate(self, **kwargs):
            return {'response': '[{"EmployeeID":"fabricated","sentiment_score":0.5,"sentiment_label":"Neutral"}]'}
    engine = NLPEngine(FakeClient())
    result = engine.analyze_sentiment(rows(1).assign(PerformanceText='Delivered the project.'))
    assert result.empty
    assert engine.get_sentiment_summary(result)['avg_sentiment'] is None


@pytest.mark.parametrize('salary', [np.nan, np.inf, 0, -100])
def test_scenarios_reject_unmeasured_or_invalid_salary(salary):
    from src.scenario_engine import ScenarioEngineError
    frame = rows()
    frame.loc[0, 'Salary'] = salary
    with pytest.raises(ScenarioEngineError, match='salaries'):
        ScenarioEngine(frame).simulate_compensation_change('percentage', {'scope': 'all'}, 10)


def test_cox_only_uses_configured_pre_outcome_covariates():
    frame = rows(60).assign(is_active=[0, 1] * 30, UndeclaredNumericProxy=range(60))
    engine = SurvivalEngine(frame)
    assert 'is_active' not in engine.available_covariates
    assert 'UndeclaredNumericProxy' not in engine.available_covariates


def test_hiring_source_missing_responses_do_not_lower_performance_rate():
    from api.routes.quality_of_hire import _safe_source
    frame = rows().assign(HireSource='Referral', LastRating=[5.] * 10 + [np.nan] * 30)
    source = QualityOfHireEngine(frame).calculate_source_effectiveness().iloc[0].to_dict()
    assert source['high_performer_rate'] == 100
    assert source['performance_observations'] == 10
    frame['LastRating'] = np.nan
    frame['Attrition'] = pd.NA
    source = QualityOfHireEngine(frame).calculate_source_effectiveness().iloc[0].to_dict()
    response = _safe_source(source)
    assert response.quality_score is None
    assert response.grade == 'Unavailable'
    assert response.avg_performance is None
    assert response.retention_rate is None
    response.model_dump_json()
