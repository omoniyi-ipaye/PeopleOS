"""
Tests for Experience Engine.
"""

import pandas as pd
import pytest
import numpy as np

from src.experience_engine import ExperienceEngine, ExperienceEngineError


@pytest.fixture
def sample_data_with_experience() -> pd.DataFrame:
    """Returns sample data with experience columns."""
    np.random.seed(42)
    n = 100

    return pd.DataFrame({
        'EmployeeID': [f'EMP{i:04d}' for i in range(1, n + 1)],
        'Dept': np.random.choice(['Engineering', 'Sales', 'HR', 'Marketing'], n),
        'Tenure': np.random.uniform(0.5, 15, n).round(1),
        'Salary': np.random.uniform(40000, 150000, n).round(0),
        'LastRating': np.random.uniform(1, 5, n).round(1),
        'Age': np.random.randint(22, 65, n),
        'eNPS_Score': np.random.randint(0, 11, n),
        'Pulse_Score': np.random.uniform(1, 5, n).round(1),
        'ManagerSatisfaction': np.random.uniform(1, 5, n).round(1),
        'WorkLifeBalance': np.random.uniform(1, 5, n).round(1),
        'CareerGrowthSatisfaction': np.random.uniform(1, 5, n).round(1),
        'YearsSinceLastPromotion': np.random.uniform(0, 8, n).round(1),
        'ManagerID': [f'MGR{np.random.randint(1, 11):02d}' for _ in range(n)],
    })


@pytest.fixture
def sample_data_minimal() -> pd.DataFrame:
    """Returns sample data without experience columns."""
    np.random.seed(42)
    n = 100

    return pd.DataFrame({
        'EmployeeID': [f'EMP{i:04d}' for i in range(1, n + 1)],
        'Dept': np.random.choice(['Engineering', 'Sales', 'HR'], n),
        'Tenure': np.random.uniform(0.5, 15, n).round(1),
        'Salary': np.random.uniform(40000, 150000, n).round(0),
        'LastRating': np.random.uniform(1, 5, n).round(1),
        'Age': np.random.randint(22, 65, n),
    })


@pytest.fixture
def sample_data_with_onboarding() -> pd.DataFrame:
    """Returns sample data with onboarding columns."""
    np.random.seed(42)
    n = 100

    return pd.DataFrame({
        'EmployeeID': [f'EMP{i:04d}' for i in range(1, n + 1)],
        'Dept': np.random.choice(['Engineering', 'Sales', 'HR'], n),
        'Tenure': np.random.uniform(0.1, 2, n).round(2),
        'Salary': np.random.uniform(40000, 100000, n).round(0),
        'LastRating': np.random.uniform(2, 5, n).round(1),
        'Age': np.random.randint(22, 45, n),
        'Onboarding_30d': np.random.uniform(2, 5, n).round(1),
        'Onboarding_60d': np.random.uniform(2, 5, n).round(1),
        'Onboarding_90d': np.random.uniform(2, 5, n).round(1),
    })


class TestExperienceEngineInitialization:
    """Tests for ExperienceEngine initialization."""

    def test_init_with_experience_columns(self, sample_data_with_experience):
        engine = ExperienceEngine(sample_data_with_experience)
        assert engine.has_enps is True
        assert engine.has_pulse is True
        assert engine.has_manager_satisfaction is True
        assert engine.has_work_life is True
        assert engine.has_career_growth is True
        assert engine.available_survey_signals >= 5

    def test_init_minimal_data(self, sample_data_minimal):
        engine = ExperienceEngine(sample_data_minimal)
        assert engine.has_enps is False
        assert engine.has_pulse is False
        assert engine.available_survey_signals == 0
        assert len(engine.warnings) > 0

    def test_init_with_onboarding(self, sample_data_with_onboarding):
        engine = ExperienceEngine(sample_data_with_onboarding)
        assert engine.has_onboarding is True
        assert engine.has_onboarding_30d is True
        assert engine.has_onboarding_60d is True
        assert engine.has_onboarding_90d is True

    def test_init_missing_employee_id(self):
        df = pd.DataFrame({'Dept': ['Engineering', 'Sales'], 'Tenure': [1.0, 2.0]})
        with pytest.raises(ExperienceEngineError, match="Missing required columns"):
            ExperienceEngine(df)

    def test_exi_computed_on_init(self, sample_data_with_experience):
        engine = ExperienceEngine(sample_data_with_experience)
        assert '_exi_score' in engine.df.columns
        assert engine.df['_exi_score'].notna().all()
        assert engine.df['_exi_score'].between(0, 100).all()


class TestExperienceIndexCalculation:
    """Tests for EXI calculation."""

    def test_calculate_experience_index(self, sample_data_with_experience):
        engine = ExperienceEngine(sample_data_with_experience)
        result = engine.calculate_experience_index()
        assert result['available'] is True
        assert 0 <= result['overall_exi'] <= 100
        assert result['total_employees'] == len(sample_data_with_experience)
        assert result['interpretation'] is not None

    def test_calculate_experience_index_by_group(self, sample_data_with_experience):
        engine = ExperienceEngine(sample_data_with_experience)
        result = engine.calculate_experience_index(group_by='Dept')
        assert result['available'] is True
        assert 'by_group' in result
        assert len(result['by_group']) > 0
        for group in result['by_group']:
            assert 'group' in group
            assert 'exi' in group
            assert 'count' in group

    def test_get_employee_exi(self, sample_data_with_experience):
        engine = ExperienceEngine(sample_data_with_experience)
        result = engine.get_employee_exi('EMP0001')
        assert result['available'] is True
        assert result['EmployeeID'] == 'EMP0001'
        assert 0 <= result['exi_score'] <= 100
        assert result['segment'] in [
            'Very high score band', 'High score band', 'Mid score band',
            'Low score band', 'Very low score band'
        ]

    def test_get_employee_exi_not_found(self, sample_data_with_experience):
        engine = ExperienceEngine(sample_data_with_experience)
        result = engine.get_employee_exi('INVALID_ID')
        assert result['available'] is False
        assert 'not found' in result['reason'].lower()


class TestEngagementSegmentation:
    """Tests for configured score-band segmentation."""

    def test_get_engagement_segments(self, sample_data_with_experience):
        engine = ExperienceEngine(sample_data_with_experience)
        result = engine.get_engagement_segments()
        assert result['available'] is True
        assert 'segments' in result
        assert len(result['segments']) == 5
        assert [s['segment'] for s in result['segments']] == [
            'Very high score band', 'High score band', 'Mid score band',
            'Low score band', 'Very low score band'
        ]

    def test_segment_percentages_are_suppression_aware(self, sample_data_with_experience):
        engine = ExperienceEngine(sample_data_with_experience)
        result = engine.get_engagement_segments()
        visible = [s for s in result['segments'] if not s.get('suppressed')]
        suppressed = [s for s in result['segments'] if s.get('suppressed')]
        assert all(s['percentage'] is not None and s['count'] is not None for s in visible)
        assert all(s['percentage'] is None and s['count'] is None and s['avg_exi'] is None for s in suppressed)
        if not result.get('suppression_applied'):
            total_pct = sum(s['percentage'] for s in visible)
            assert 99 <= total_pct <= 101

    def test_health_indicator(self, sample_data_with_experience):
        engine = ExperienceEngine(sample_data_with_experience)
        result = engine.get_engagement_segments()
        assert result['health_indicator'] == 'Measured score distribution'


class TestExperienceDrivers:
    """Tests for experience association analysis."""

    def test_identify_drivers(self, sample_data_with_experience):
        engine = ExperienceEngine(sample_data_with_experience)
        result = engine.identify_experience_drivers()
        assert result['available'] is True
        assert 'drivers' in result
        assert len(result['drivers']) > 0

    def test_driver_structure(self, sample_data_with_experience):
        engine = ExperienceEngine(sample_data_with_experience)
        result = engine.identify_experience_drivers()
        for driver in result['drivers']:
            assert 'factor' in driver
            assert 'correlation' in driver
            assert 'impact' in driver
            assert driver['impact'] in ['High', 'Medium', 'Low']
            assert driver['metric_semantics'] == 'observational_association_excluding_index_components'


class TestAtRiskEmployees:
    """Tests aggregate low-score monitoring; no individual risk list is exposed."""

    def test_get_at_risk(self, sample_data_with_experience):
        engine = ExperienceEngine(sample_data_with_experience)
        result = engine.get_at_risk_employees()
        assert result['available'] is True
        assert result['employees'] is None
        assert 'total_at_risk' in result
        assert result['metric_semantics'] == 'aggregate_count_below_configured_composite_threshold_not_employee_risk_prediction'

    def test_at_risk_output_is_aggregate_only(self, sample_data_with_experience):
        engine = ExperienceEngine(sample_data_with_experience)
        result = engine.get_at_risk_employees(limit=5)
        assert result['employees'] is None
        assert isinstance(result['by_department'], list)
        assert all('EmployeeID' not in row for row in result['by_department'])

    def test_at_risk_custom_threshold(self, sample_data_with_experience):
        engine = ExperienceEngine(sample_data_with_experience)
        result = engine.get_at_risk_employees(threshold=60)
        assert result['threshold_used'] == 60
        assert result['employees'] is None
        assert result['total_at_risk'] is None or isinstance(result['total_at_risk'], int)


class TestLifecycleAnalysis:
    """Tests for lifecycle experience analysis."""

    def test_get_lifecycle_experience(self, sample_data_with_experience):
        engine = ExperienceEngine(sample_data_with_experience)
        result = engine.get_lifecycle_experience()
        assert result['available'] is True
        assert 'stages' in result

    def test_lifecycle_stages_present(self, sample_data_with_experience):
        engine = ExperienceEngine(sample_data_with_experience)
        result = engine.get_lifecycle_experience()
        stage_names = [s['stage'] for s in result['stages']]
        assert len(stage_names) > 0
        assert all(s['respondent_count'] >= 10 for s in result['stages'])


class TestManagerImpact:
    """Manager-level experience ranking is intentionally disabled."""

    def test_analyze_manager_impact(self, sample_data_with_experience):
        engine = ExperienceEngine(sample_data_with_experience)
        result = engine.analyze_manager_impact()
        assert result['available'] is False
        assert 'disabled' in result['reason'].lower()
        assert result['bottom_managers'] is None
        assert result['top_managers'] is None

    def test_manager_impact_without_manager_id(self, sample_data_minimal):
        engine = ExperienceEngine(sample_data_minimal)
        result = engine.analyze_manager_impact()
        assert result['available'] is False
        assert 'disabled' in result['reason'].lower()


class TestAvailableSignals:
    """Tests for available signals reporting."""

    def test_get_signals_with_data(self, sample_data_with_experience):
        engine = ExperienceEngine(sample_data_with_experience)
        result = engine.get_available_signals()
        assert result['has_enps'] is True
        assert result['has_pulse'] is True
        assert result['total_signals'] >= 5

    def test_get_signals_without_data(self, sample_data_minimal):
        engine = ExperienceEngine(sample_data_minimal)
        result = engine.get_available_signals()
        assert result['has_enps'] is False
        assert result['has_pulse'] is False
        assert result['total_signals'] == 0
        assert len(result['recommendations']) > 0


class TestAnalyzeAll:
    """Tests for comprehensive analysis."""

    def test_analyze_all(self, sample_data_with_experience):
        engine = ExperienceEngine(sample_data_with_experience)
        result = engine.analyze_all()
        assert 'experience_index' in result
        assert 'segments' in result
        assert 'drivers' in result
        assert 'at_risk' in result
        assert 'lifecycle' in result
        assert 'manager_impact' in result
        assert 'signals' in result
        assert 'summary' in result
        assert 'recommendations' in result
        assert 'warnings' in result

    def test_analyze_all_summary(self, sample_data_with_experience):
        engine = ExperienceEngine(sample_data_with_experience)
        result = engine.analyze_all()
        summary = result['summary']
        assert 'overall_exi' in summary
        assert 'health_indicator' in summary
        assert 'total_employees' in summary
        assert result['at_risk']['employees'] is None
        assert result['manager_impact']['available'] is False


class TestEdgeCases:
    """Tests for edge cases."""

    def test_small_dataset(self):
        df = pd.DataFrame({
            'EmployeeID': [f'EMP{i}' for i in range(15)],
            'Dept': ['Engineering'] * 15,
            'Tenure': [1.0] * 15,
            'Salary': [50000] * 15,
            'LastRating': [3.0] * 15,
            'Age': [30] * 15,
        })
        engine = ExperienceEngine(df)
        assert len(engine.warnings) > 0

    def test_missing_values_handled(self, sample_data_with_experience):
        df = sample_data_with_experience.copy()
        df.loc[0, 'eNPS_Score'] = None
        df.loc[1, 'Pulse_Score'] = None
        engine = ExperienceEngine(df)
        result = engine.calculate_experience_index()
        assert result['available'] is True
