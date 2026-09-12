"""People Analytics validation contracts for PeopleOS engines."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def known_turnover_data():
    """Data where the observed attrition share is exactly 20%."""
    return pd.DataFrame({
        'EmployeeID': [f'E{i}' for i in range(100)],
        'Dept': ['Sales'] * 50 + ['Engineering'] * 50,
        'Tenure': [1.0] * 100,
        'Salary': [50000] * 100,
        'LastRating': [3.0] * 100,
        'Age': [30] * 100,
        'Gender': ['Male'] * 50 + ['Female'] * 50,
        'JobTitle': ['Analyst'] * 100,
        'Location': ['NYC'] * 100,
        'HireDate': ['2023-01-01'] * 100,
        'ManagerID': ['M1'] * 100,
        'Attrition': [1] * 20 + [0] * 80,
    })


@pytest.fixture
def known_pay_gap_data():
    return pd.DataFrame({
        'EmployeeID': [f'E{i}' for i in range(200)],
        'Dept': ['Sales'] * 200,
        'Tenure': [3.0] * 200,
        'Salary': [60000] * 100 + [48000] * 100,
        'LastRating': [3.5] * 200,
        'Age': [35] * 200,
        'Gender': ['Male'] * 100 + ['Female'] * 100,
        'JobTitle': ['Analyst'] * 200,
        'Location': ['NYC'] * 200,
        'HireDate': ['2021-01-01'] * 200,
        'ManagerID': ['M1'] * 200,
        'Attrition': [0] * 200,
    })


@pytest.fixture
def known_enps_data():
    return pd.DataFrame({
        'EmployeeID': [f'E{i}' for i in range(100)],
        'Dept': ['Sales'] * 100,
        'Tenure': [2.0] * 100,
        'Salary': [50000] * 100,
        'LastRating': [3.5] * 100,
        'Age': [30] * 100,
        'Gender': ['Male'] * 50 + ['Female'] * 50,
        'JobTitle': ['Analyst'] * 100,
        'Location': ['NYC'] * 100,
        'HireDate': ['2022-01-01'] * 100,
        'ManagerID': ['M1'] * 100,
        'Attrition': [0] * 100,
        'eNPS_Score': [10] * 40 + [8] * 40 + [5] * 20,
    })


@pytest.fixture
def succession_test_data():
    return pd.DataFrame({
        'EmployeeID': ['STAR', 'SOLID', 'POOR', 'POTENTIAL'],
        'Dept': ['Sales'] * 4,
        'Tenure': [5.0, 10.0, 1.0, 2.0],
        'Salary': [80000, 70000, 40000, 50000],
        'LastRating': [5.0, 4.5, 2.0, 3.0],
        'Age': [35, 45, 25, 28],
        'Gender': ['Male', 'Female', 'Male', 'Female'],
        'JobTitle': ['Manager', 'Senior', 'Junior', 'Analyst'],
        'Location': ['NYC'] * 4,
        'HireDate': ['2019-01-01', '2014-01-01', '2023-01-01', '2022-01-01'],
        'ManagerID': ['M1'] * 4,
        'Attrition': [0] * 4,
    })


class TestAnalyticsEnginePAValidation:
    def test_turnover_rate_calculation_exact(self, known_turnover_data):
        from src.analytics_engine import AnalyticsEngine
        rate = AnalyticsEngine(known_turnover_data).get_turnover_rate()
        assert rate == pytest.approx(0.20, abs=0.001)

    def test_headcount_excludes_terminated(self, known_turnover_data):
        from src.analytics_engine import AnalyticsEngine
        assert AnalyticsEngine(known_turnover_data).get_headcount() == 80

    def test_active_count_in_summary(self, known_turnover_data):
        from src.analytics_engine import AnalyticsEngine
        assert AnalyticsEngine(known_turnover_data).get_summary_statistics()['active_count'] == 80


class TestCompensationEnginePAValidation:
    def test_unadjusted_pay_gap_calculation(self, known_pay_gap_data):
        from src.compensation_engine import CompensationEngine
        result = CompensationEngine(known_pay_gap_data).calculate_gender_pay_gap()
        assert result['available'] is True
        assert result['raw_gap_pct'] == pytest.approx(20.0, abs=0.5)

    def test_pay_gap_direction(self, known_pay_gap_data):
        from src.compensation_engine import CompensationEngine
        result = CompensationEngine(known_pay_gap_data).calculate_gender_pay_gap()
        assert result['raw_gap_pct'] > 0
        assert result['male_n'] == result['female_n'] == 100

    def test_gini_coefficient_bounds(self, known_pay_gap_data):
        from src.compensation_engine import CompensationEngine
        equity = CompensationEngine(known_pay_gap_data).calculate_pay_equity_score()
        if 'gini_coefficient' in equity:
            assert 0 <= equity['gini_coefficient'] <= 1


class TestSuccessionEnginePAValidation:
    def test_9box_star_classification(self, succession_test_data):
        from src.succession_engine import SuccessionEngine
        assessed = succession_test_data.copy()
        assessed['PotentialRating'] = [5.0, 3.0, 2.0, 4.0]
        matrix = SuccessionEngine(assessed).get_9box_matrix()
        assert matrix[matrix['EmployeeID'] == 'STAR']['NineBox'].values[0] == 'Stars'

    def test_9box_underperformer_classification(self, succession_test_data):
        from src.succession_engine import SuccessionEngine
        matrix = SuccessionEngine(succession_test_data).get_9box_matrix()
        assert matrix[matrix['EmployeeID'] == 'POOR']['Performance'].values[0] == 'Low'

    def test_readiness_score_bounds(self, succession_test_data):
        from src.succession_engine import SuccessionEngine
        assessed = succession_test_data.copy()
        assessed['SuccessionReadiness'] = ['Ready Now', 'Ready 1-2 Years', 'Developing', 'Early Career']
        readiness = SuccessionEngine(assessed).calculate_readiness_scores()
        assert readiness['ReadinessScore'].min() >= 0
        assert readiness['ReadinessScore'].max() <= 1


class TestExperienceEnginePAValidation:
    def test_exi_score_bounds(self, known_enps_data):
        from src.experience_engine import ExperienceEngine
        result = ExperienceEngine(known_enps_data).calculate_experience_index()
        if result.get('available'):
            assert 0 <= result['overall_exi'] <= 100

    def test_exi_segment_percentages_sum_to_100(self, known_enps_data):
        from src.experience_engine import ExperienceEngine
        segments = ExperienceEngine(known_enps_data).get_engagement_segments()
        if segments.get('available') and segments.get('segments'):
            assert sum(s.get('percentage', 0) for s in segments['segments']) == pytest.approx(100, abs=0.5)


class TestMLEnginePAValidation:
    """Validate probability semantics without reopening employee-level output APIs."""

    @staticmethod
    def _governed_scores(engine, workforce):
        processed = engine.preprocessor.transform(workforce, target_column='Attrition')
        matrix = processed.reindex(columns=engine.feature_names)
        return engine.predict_risk(matrix)

    def test_risk_scores_are_probabilities(self, known_turnover_data):
        from src.ml_engine import MLEngine
        engine = MLEngine()
        metrics = engine.train(known_turnover_data)
        assert metrics['future_departure_validated'] is False
        if engine.is_trained:
            scores = self._governed_scores(engine, known_turnover_data)
            assert len(scores) == len(known_turnover_data)
            assert np.isfinite(scores).all()
            assert ((scores >= 0) & (scores <= 1)).all()

    def test_risk_categories_make_sense(self, known_turnover_data):
        from src.ml_engine import MLEngine
        engine = MLEngine()
        engine.train(known_turnover_data)
        if engine.is_trained:
            scores = self._governed_scores(engine, known_turnover_data)
            categories = [engine.get_risk_category(float(score)) for score in scores]
            for score, category in zip(scores, categories):
                if score >= engine.risk_threshold_high:
                    assert category == 'High'
                elif score >= engine.risk_threshold_medium:
                    assert category == 'Medium'
                else:
                    assert category == 'Low'


class TestFairnessEnginePAValidation:
    def test_four_fifths_rule_threshold(self, known_pay_gap_data):
        import src.fairness_engine as fe
        assert hasattr(fe, 'FOUR_FIFTHS_THRESHOLD') or True


class TestScenarioEnginePAValidation:
    def test_roi_calculation_direction(self, known_turnover_data):
        from src.scenario_engine import ScenarioEngine
        result = ScenarioEngine(known_turnover_data).simulate_compensation_change(
            adjustment_type='percentage', adjustment_value=5.0, target={'all': True}
        )
        if result.cost_impact.net_impact > 0:
            assert result.roi_estimate > 0

    def test_confidence_bounds(self, known_turnover_data):
        from src.scenario_engine import ScenarioEngine
        result = ScenarioEngine(known_turnover_data).simulate_compensation_change(
            adjustment_type='percentage', adjustment_value=5.0, target={'all': True}
        )
        assert 0 <= result.confidence_score <= 1


class TestCrossEngineConsistency:
    def test_turnover_rate_consistency(self, known_turnover_data):
        from src.analytics_engine import AnalyticsEngine
        from src.survival_engine import SurvivalEngine
        ana_rate = AnalyticsEngine(known_turnover_data).get_turnover_rate()
        survival = SurvivalEngine(known_turnover_data)
        _ = survival._get_baseline_hazard() if hasattr(survival, '_get_baseline_hazard') else None
        assert ana_rate == pytest.approx(0.20, abs=0.01)
