"""Known-answer contracts for the specialist's measurement repairs.

These are correctness regressions, distinct from the original defect
reproductions. All employee and survey records are synthetic.
"""
import numpy as np
import pandas as pd
import pytest

from src.compensation_engine import CompensationEngine
from src.quality_of_hire_engine import QualityOfHireEngine
from src.scenario_engine import ScenarioEngine
from src.sentiment_engine import SentimentEngine
from src.team_dynamics_engine import TeamDynamicsEngine


def people(n=20):
    return pd.DataFrame({
        'EmployeeID': [f'E{i}' for i in range(n)], 'Dept': 'A',
        'Salary': 60000., 'Tenure': 2., 'LastRating': 4., 'Attrition': 0,
        'Age': 30, 'HireSource': 'Referral', 'HireDate': '2024-01-01',
    })


def test_missing_department_scenario_serializes_and_remains_targetable():
    from api.routes.scenario import _convert_result
    frame=people().assign(Dept=[None,'','  ']+['001']*17)
    engine=ScenarioEngine(frame)
    result=engine.simulate_compensation_change('percentage',{'scope':'all'},10,6)
    response=_convert_result(result)
    assert set(response.affected_departments)=={'Unknown','001'}
    assert response.cost_impact.salary_change==60000
    unknown=engine.simulate_compensation_change('percentage',{'scope':'department','department':'Unknown'},10,6)
    assert unknown.affected_employees==3
    assert unknown.cost_impact.salary_change==9000


def test_succession_unknown_department_preserves_unassessed_people():
    from src.succession_engine import SuccessionEngine
    frame=people().assign(Dept=[None,'','  ']+['001']*17)
    bench=SuccessionEngine(frame).calculate_bench_strength().set_index('Dept')
    assert bench.Total.sum()==20
    assert bench.loc['Unknown','Unassessed']==3
    assert pd.isna(bench.loc['Unknown','BenchStrength'])


def test_full_sentiment_response_has_no_nonfinite_values_before_serialization():
    import asyncio
    import json
    from types import SimpleNamespace
    from api.routes.sentiment import get_sentiment_analysis
    frame=people().assign(Dept=[None]*10+['001']*10)
    survey=pd.DataFrame({'EmployeeID':frame.EmployeeID,'SurveyDate':'2026-01-01',
                         'eNPSScore':[10]*10+[0]*10})
    state=SimpleNamespace(sentiment_engine=SentimentEngine(frame,enps_df=survey))
    response=asyncio.run(get_sentiment_analysis(state))
    payload=response.model_dump(mode='python')
    json.dumps(payload,allow_nan=False)
    assert payload['enps']['overall_enps']==0


def test_survey_population_and_score_exclusions_are_counted_separately():
    survey = pd.DataFrame({
        'EmployeeID': ['E0', 'E1', 'E2', 'OTHER', None, ''],
        'eNPSScore': [10, 0, 11, 10, 10, 10],
    })
    result = SentimentEngine(people(), enps_df=survey).calculate_enps()
    assert result['overall_enps'] == 0  # One promoter and one detractor.
    assert result['total_responses'] == 2
    assert result['survey_coverage']['enps'] == {
        'input_rows': 6, 'matched_rows': 3, 'unmatched_rows': 1,
        'missing_employee_id_rows': 2, 'valid_score_rows': 2, 'invalid_score_rows': 1,
    }


@pytest.mark.parametrize('survey', [
    pd.DataFrame({'EmployeeID': ['OTHER'], 'eNPSScore': [10]}),
    pd.DataFrame({'eNPSScore': [10]}),
])
def test_unmatched_or_unidentified_surveys_cannot_supply_an_enps(survey):
    result = SentimentEngine(people(), enps_df=survey).calculate_enps()
    assert result['available'] is False
    assert result['survey_coverage']['enps']['matched_rows'] == 0
    assert 'overall_enps' not in result


def test_survey_join_does_not_guess_identifier_normalization():
    population = people().assign(EmployeeID=[f'{i:04d}' for i in range(20)])
    survey = pd.DataFrame({'EmployeeID': ['0001', 1], 'eNPSScore': [10, 0]})
    result = SentimentEngine(population, enps_df=survey).calculate_enps()
    assert result['total_responses'] == 1
    assert result['overall_enps'] == 100
    assert result['survey_coverage']['enps']['unmatched_rows'] == 1


def test_unmatched_onboarding_is_unavailable_not_zero_observed_flags():
    survey = pd.DataFrame({'EmployeeID': ['OTHER'], 'SurveyType': ['30-day'], 'OverallScore': [1]})
    engine = SentimentEngine(people(), onboarding_df=survey)
    assert engine.get_onboarding_health()['available'] is False
    warnings = engine.detect_early_warnings()
    assert warnings['available'] is False
    assert warnings['survey_coverage']['onboarding']['unmatched_rows'] == 1
    summary = engine.analyze_all()['summary']
    assert summary['employees_at_risk'] is None
    assert summary['survey_flags_available'] is False


def test_onboarding_latest_response_is_shared_by_health_and_trajectory():
    survey = pd.DataFrame({
        'EmployeeID': ['E0', 'E0', 'OTHER', 'E0'],
        'SurveyType': ['30-day', '30-day', '30-day', 'unsupported'],
        'SurveyDate': ['2026-02-01', '2026-01-01', '2026-02-01', '2026-03-01'],
        'OverallScore': [5, 1, 1, 1], 'ManagerSupport': [5, 1, 1, 1],
    })
    engine = SentimentEngine(people(), onboarding_df=survey)
    trajectory, health = engine.analyze_onboarding_trajectory(), engine.get_onboarding_health()
    assert trajectory['trajectories'][0]['scores'] == {'30-day': 5}
    assert health['by_survey_type'] == [
        {'survey_type': '30-day', 'avg_score': 5, 'responses': 1, 'healthy_pct': 100},
    ]
    assert health['dimension_scores'] == [{'dimension': 'ManagerSupport', 'avg_score': 5}]
    assert health['survey_coverage']['onboarding'] == {
        'input_rows': 4, 'matched_rows': 3, 'unmatched_rows': 1,
        'missing_employee_id_rows': 0, 'unsupported_type_rows': 1,
        'superseded_rows': 1, 'latest_response_rows': 1, 'valid_overall_score_rows': 1,
    }


def test_latest_missing_score_does_not_resurrect_superseded_measurement():
    survey = pd.DataFrame({
        'EmployeeID': ['E0', 'E0'], 'SurveyType': ['30-day'] * 2,
        'SurveyDate': ['2026-01-01', '2026-02-01'], 'OverallScore': [5, np.nan],
    })
    engine = SentimentEngine(people(), onboarding_df=survey)
    assert engine.analyze_onboarding_trajectory()['trajectories'][0]['scores'] == {}
    assert engine.get_onboarding_health()['by_survey_type'] == []


def test_no_date_onboarding_tie_uses_explicit_input_order():
    survey = pd.DataFrame({'EmployeeID': ['E0'] * 2, 'SurveyType': ['30-day'] * 2,
                           'OverallScore': [1, 5]})
    result = SentimentEngine(people(), onboarding_df=survey).get_onboarding_health()
    assert result['by_survey_type'][0]['avg_score'] == 5
    assert 'input_order' in result['response_selection']


def test_onboarding_warning_total_precedes_top_ten_display_and_deduplicates_people():
    frame = people(15)
    survey = pd.DataFrame({'EmployeeID': frame.EmployeeID, 'SurveyType': '30-day',
                           'SurveyDate': '2026-01-01', 'OverallScore': 2})
    enps = pd.DataFrame({'EmployeeID': ['E0'], 'eNPSScore': [0], 'SurveyDate': ['2026-01-01']})
    engine = SentimentEngine(frame, enps_df=enps, onboarding_df=survey)
    trajectory, warnings = engine.analyze_onboarding_trajectory(), engine.detect_early_warnings()
    assert trajectory['summary']['at_risk_count'] == 15
    assert len(trajectory['at_risk_employees']) == 10
    assert warnings['summary']['total_at_risk'] == 15
    assert len({row['EmployeeID'] for row in warnings['warnings']}) == 15


def test_single_measured_hire_cannot_receive_a_source_grade():
    frame = people().assign(LastRating=[5] + [None] * 19, Attrition=None)
    row = QualityOfHireEngine(frame).calculate_source_effectiveness().iloc[0]
    assert pd.isna(row['quality_score'])
    assert row['grade'] == 'Unavailable'
    assert row['component_observations']['performance'] == 1
    assert row['component_coverage']['performance'] == .05
    assert row['quality_unavailable_reason']


def test_sources_share_weights_and_insufficient_components_suppress_comparison():
    frame = people(40)
    frame['HireSource'] = ['Measured'] * 20 + ['Unknown outcomes'] * 20
    frame['Attrition'] = [0] * 20 + [None] * 20
    rows = QualityOfHireEngine(frame).calculate_source_effectiveness().set_index('HireSource')
    measured, unknown = rows.loc['Measured'], rows.loc['Unknown outcomes']
    assert measured['quality_score'] == 85.7
    assert pd.isna(unknown['quality_score'])
    assert measured['quality_weights'] == unknown['quality_weights']
    assert measured['quality_weights'] == pytest.approx({'performance': 4 / 7, 'retention': 3 / 7})
    assert 'retention' in unknown['quality_unavailable_reason']
    assert unknown['outcome_observations'] == 0


def test_wholly_unmeasured_dataset_components_are_excluded_once():
    frame = people(40).assign(Attrition=None)
    frame['HireSource'] = ['A'] * 20 + ['B'] * 20
    rows = QualityOfHireEngine(frame).calculate_source_effectiveness()
    assert rows.quality_score.tolist() == [75, 75]
    assert all(value == {'performance': 1.0} for value in rows.quality_weights)
    assert all(value == ['performance'] for value in rows.quality_components)


def test_source_measured_minimum_boundary_is_component_specific():
    frame = people(20).assign(Attrition=None)
    frame['HireSource'] = ['A'] * 10 + ['B'] * 10
    frame.loc[19, 'LastRating'] = np.nan
    rows = QualityOfHireEngine(frame).calculate_source_effectiveness().set_index('HireSource')
    assert rows.loc['A', 'quality_score'] == 75
    assert pd.isna(rows.loc['B', 'quality_score'])
    assert rows.loc['B', 'component_observations']['performance'] == 9


def test_sparse_outcomes_do_not_reappear_as_source_flags_or_relative_rating_claims():
    frame = people().assign(LastRating=[5] + [None] * 19, Attrition=[1] + [None] * 19)
    insights = QualityOfHireEngine(frame).get_hiring_insights()
    assert insights['top_sources'] == []
    assert insights['red_flags'] == []
    assert insights['roi_analysis'] == {}


def test_quality_api_preserves_comparison_and_coverage_contract():
    from api.routes.quality_of_hire import _safe_source
    row = QualityOfHireEngine(people()).calculate_source_effectiveness().iloc[0].to_dict()
    response = _safe_source(row).model_dump()
    for key in ['quality_components', 'quality_weights', 'component_observations',
                'component_coverage', 'minimum_component_observations', 'outcome_observations']:
        assert response[key] == row[key]


def test_loss_making_expansion_has_no_supported_payback():
    result = ScenarioEngine(people()).simulate_headcount_change('expansion', {'scope': 'all'}, change_count=1)
    assert result.cost_impact.net_impact < 0
    assert result.payback_months is None
    assert any('multi-period cash-flow' in assumption for assumption in result.assumptions)


def test_reduction_severance_payback_is_three_months_not_four():
    result = ScenarioEngine(people()).simulate_headcount_change('reduction', {'scope': 'all'}, change_count=2)
    assert result.cost_impact.total_cost == 30000
    assert result.cost_impact.total_benefit == 120000
    assert result.payback_months == 3  # 30,000 / (120,000 / 12).


def test_unknown_department_is_preserved_in_composition_and_outcome_metrics():
    frame = people()
    frame.loc[:4, 'Dept'] = None
    frame.loc[5:9, 'Dept'] = ' '
    frame.loc[0, 'Attrition'] = 1
    engine = TeamDynamicsEngine(frame)
    composition = engine.get_team_composition().set_index('Dept')
    assert composition.Headcount.sum() == 19
    assert composition.loc['Unknown', 'Headcount'] == 9
    health = engine.calculate_team_health_scores().set_index('Dept')
    assert health.loc['Unknown', 'AttritionRate'] == 10
    assert health.loc['Unknown', 'Headcount'] == 9


def test_invalid_compa_ratios_remain_unavailable_and_valid_thresholds_survive():
    frame = people(8).assign(CompaRatio=[np.nan, np.inf, 0, -1, .75, .8, 1.2, 1.25])
    rows = CompensationEngine(frame).calculate_compa_ratio()
    assert rows.CompaStatus.tolist() == ['Unavailable'] * 4 + [
        'Below reference', 'Near reference', 'Near reference', 'Above reference',
    ]
    assert rows.CompaRatio.iloc[:4].isna().all()
    assert rows.BandMidpoint.iloc[:4].isna().all()
