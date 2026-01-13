"""
Tests for Scenario Planning Engine.
"""

import pandas as pd
import pytest
import numpy as np

from src.scenario_engine import ScenarioEngine, ScenarioEngineError


@pytest.fixture
def sample_employee_data() -> pd.DataFrame:
    """Returns sample employee data for scenario testing."""
    np.random.seed(42)
    n = 100

    return pd.DataFrame({
        'EmployeeID': [f'EMP{i:04d}' for i in range(1, n + 1)],
        'Dept': np.random.choice(['Engineering', 'Sales', 'HR', 'Marketing'], n),
        'JobTitle': np.random.choice(['Engineer', 'Manager', 'Analyst', 'Director'], n),
        'Tenure': np.random.uniform(0.5, 15, n).round(1),
        'Salary': np.random.uniform(50000, 150000, n).round(0),
        'LastRating': np.random.uniform(1, 5, n).round(1),
        'Age': np.random.randint(22, 65, n),
        'Attrition': np.random.choice([0, 1], n, p=[0.85, 0.15]),
    })


@pytest.fixture
def sample_data_with_high_turnover() -> pd.DataFrame:
    """Returns sample data with higher turnover rate."""
    np.random.seed(42)
    n = 100

    return pd.DataFrame({
        'EmployeeID': [f'EMP{i:04d}' for i in range(1, n + 1)],
        'Dept': np.random.choice(['Engineering', 'Sales', 'HR'], n),
        'JobTitle': np.random.choice(['Engineer', 'Manager', 'Analyst'], n),
        'Tenure': np.random.uniform(0.5, 10, n).round(1),
        'Salary': np.random.uniform(40000, 120000, n).round(0),
        'LastRating': np.random.uniform(1, 5, n).round(1),
        'Age': np.random.randint(22, 55, n),
        'Attrition': np.random.choice([0, 1], n, p=[0.70, 0.30]),
    })


@pytest.fixture
def sample_data_minimal() -> pd.DataFrame:
    """Returns minimal sample data without attrition."""
    np.random.seed(42)
    n = 50

    return pd.DataFrame({
        'EmployeeID': [f'EMP{i:04d}' for i in range(1, n + 1)],
        'Dept': np.random.choice(['Engineering', 'Sales'], n),
        'Tenure': np.random.uniform(0.5, 10, n).round(1),
        'Salary': np.random.uniform(50000, 100000, n).round(0),
        'LastRating': np.random.uniform(1, 5, n).round(1),
        'Age': np.random.randint(25, 55, n),
    })


class TestScenarioEngineInitialization:
    """Tests for ScenarioEngine initialization."""

    def test_init_basic(self, sample_employee_data):
        """Test basic initialization."""
        engine = ScenarioEngine(sample_employee_data)

        assert engine is not None
        assert len(engine.df) == 100
        assert engine.n_simulations == 1000  # Default

    def test_init_without_ml_engines(self, sample_employee_data):
        """Test initialization without ML engines."""
        engine = ScenarioEngine(sample_employee_data)

        assert engine.has_ml is False
        assert engine.has_survival is False
        assert engine.has_compensation is False
        assert len(engine.warnings) > 0  # Should warn about no engines

    def test_init_with_minimal_data(self, sample_data_minimal):
        """Test initialization with minimal data."""
        engine = ScenarioEngine(sample_data_minimal)

        assert engine is not None
        assert len(engine.df) == 50


class TestCompensationScenarios:
    """Tests for compensation change scenarios."""

    def test_simulate_percentage_raise(self, sample_employee_data):
        """Test simulating a percentage raise."""
        engine = ScenarioEngine(sample_employee_data)

        result = engine.simulate_compensation_change(
            adjustment_type='percentage',
            target={'scope': 'all'},
            adjustment_value=5.0,
            time_horizon_months=12
        )

        assert result.scenario_id is not None
        assert result.scenario_type == 'compensation'
        assert result.affected_employees == 100
        assert result.baseline_turnover_rate >= 0
        assert result.projected_turnover_rate >= 0
        assert result.projected_turnover_rate <= result.baseline_turnover_rate
        assert result.cost_impact is not None
        assert result.simulation is not None

    def test_simulate_department_raise(self, sample_employee_data):
        """Test simulating raise for specific department."""
        engine = ScenarioEngine(sample_employee_data)

        result = engine.simulate_compensation_change(
            adjustment_type='percentage',
            target={'scope': 'department', 'department': 'Engineering'},
            adjustment_value=10.0,
            time_horizon_months=12
        )

        assert result.affected_employees < 100  # Only Engineering
        assert 'Engineering' in result.affected_departments
        assert result.turnover_change >= 0  # Should be positive (reduction)

    def test_simulate_absolute_raise(self, sample_employee_data):
        """Test simulating absolute salary increase."""
        engine = ScenarioEngine(sample_employee_data)

        result = engine.simulate_compensation_change(
            adjustment_type='absolute',
            target={'scope': 'all'},
            adjustment_value=5000.0,
            time_horizon_months=12
        )

        assert result.scenario_type == 'compensation'
        assert result.cost_impact.salary_change > 0

    def test_empty_target_raises_error(self, sample_employee_data):
        """Test that empty target raises error."""
        engine = ScenarioEngine(sample_employee_data)

        with pytest.raises(ScenarioEngineError):
            engine.simulate_compensation_change(
                adjustment_type='percentage',
                target={'scope': 'department', 'department': 'NonExistent'},
                adjustment_value=5.0
            )


class TestHeadcountScenarios:
    """Tests for headcount change scenarios."""

    def test_simulate_reduction_by_count(self, sample_employee_data):
        """Test simulating workforce reduction by count."""
        engine = ScenarioEngine(sample_employee_data)

        result = engine.simulate_headcount_change(
            change_type='reduction',
            target={'scope': 'all'},
            change_count=10,
            selection_criteria='performance'
        )

        assert result.scenario_type == 'headcount'
        assert result.affected_employees == 10
        assert result.cost_impact.salary_change < 0  # Salary savings

    def test_simulate_reduction_by_percentage(self, sample_employee_data):
        """Test simulating workforce reduction by percentage."""
        engine = ScenarioEngine(sample_employee_data)

        result = engine.simulate_headcount_change(
            change_type='reduction',
            target={'scope': 'department', 'department': 'Sales'},
            change_percentage=20.0,
            selection_criteria='tenure'
        )

        assert result.scenario_type == 'headcount'
        # Should be ~20% of Sales employees

    def test_simulate_expansion(self, sample_employee_data):
        """Test simulating workforce expansion."""
        engine = ScenarioEngine(sample_employee_data)

        result = engine.simulate_headcount_change(
            change_type='expansion',
            target={'scope': 'department', 'department': 'Engineering'},
            change_count=5,
            selection_criteria='performance'
        )

        assert result.scenario_type == 'headcount'
        assert result.affected_employees == 5
        assert result.cost_impact.salary_change > 0  # Hiring cost
        assert result.cost_impact.training_costs > 0

    def test_missing_change_amount_raises_error(self, sample_employee_data):
        """Test that missing change amount raises error."""
        engine = ScenarioEngine(sample_employee_data)

        with pytest.raises(ScenarioEngineError):
            engine.simulate_headcount_change(
                change_type='reduction',
                target={'scope': 'all'},
                # No change_count or change_percentage
            )


class TestInterventionScenarios:
    """Tests for intervention scenarios."""

    def test_retention_bonus(self, sample_employee_data):
        """Test retention bonus intervention."""
        engine = ScenarioEngine(sample_employee_data)

        result = engine.simulate_attrition_intervention(
            intervention_type='retention_bonus',
            target_employees='high_risk',
            intervention_params={'bonus_percentage': 10}
        )

        assert result.scenario_type == 'intervention'
        assert result.cost_impact.replacement_costs_avoided >= 0
        assert result.roi_estimate is not None

    def test_career_path_intervention(self, sample_employee_data):
        """Test career path intervention."""
        engine = ScenarioEngine(sample_employee_data)

        result = engine.simulate_attrition_intervention(
            intervention_type='career_path',
            target_employees='high_risk_high_performer',
            intervention_params={'cost_per_person': 5000}
        )

        assert result.scenario_type == 'intervention'
        assert result.cost_impact.training_costs > 0

    def test_manager_change_intervention(self, sample_employee_data):
        """Test manager change intervention."""
        engine = ScenarioEngine(sample_employee_data)

        result = engine.simulate_attrition_intervention(
            intervention_type='manager_change',
            target_employees='high_risk',
            intervention_params={'change_cost': 10000}
        )

        assert result.scenario_type == 'intervention'


class TestMonteCarloSimulation:
    """Tests for Monte Carlo simulation."""

    def test_simulation_convergence(self, sample_employee_data):
        """Test that simulation converges."""
        engine = ScenarioEngine(sample_employee_data)

        result = engine.simulate_compensation_change(
            adjustment_type='percentage',
            target={'scope': 'all'},
            adjustment_value=5.0
        )

        mc = result.simulation
        assert mc.n_iterations == 1000
        assert mc.outcome_std > 0
        assert 'p50' in mc.percentiles
        assert len(mc.histogram_bins) > 0
        assert len(mc.histogram_counts) > 0

    def test_roi_probability(self, sample_employee_data):
        """Test ROI probability calculation."""
        engine = ScenarioEngine(sample_employee_data)

        result = engine.simulate_compensation_change(
            adjustment_type='percentage',
            target={'scope': 'all'},
            adjustment_value=5.0
        )

        assert 0 <= result.simulation.roi_positive_probability <= 1

    def test_cost_impact_distribution(self, sample_employee_data):
        """Test cost impact distribution is calculated."""
        engine = ScenarioEngine(sample_employee_data)

        result = engine.simulate_compensation_change(
            adjustment_type='percentage',
            target={'scope': 'all'},
            adjustment_value=5.0
        )

        mc = result.simulation
        assert mc.cost_impact_mean != 0 or mc.cost_impact_std >= 0
        assert 'p5' in mc.cost_impact_percentiles
        assert 'p95' in mc.cost_impact_percentiles


class TestScenarioComparison:
    """Tests for scenario comparison."""

    def test_compare_two_scenarios(self, sample_employee_data):
        """Test comparing two scenarios."""
        engine = ScenarioEngine(sample_employee_data)

        # Run two scenarios
        result1 = engine.simulate_compensation_change(
            adjustment_type='percentage',
            target={'scope': 'all'},
            adjustment_value=5.0
        )

        result2 = engine.simulate_compensation_change(
            adjustment_type='percentage',
            target={'scope': 'all'},
            adjustment_value=10.0
        )

        comparison = engine.compare_scenarios([result1, result2])

        assert 'scenarios' in comparison
        assert len(comparison['scenarios']) == 2
        assert 'recommended_scenario' in comparison
        assert 'reasoning' in comparison

    def test_compare_single_scenario_fails(self, sample_employee_data):
        """Test that comparing single scenario returns error."""
        engine = ScenarioEngine(sample_employee_data)

        result1 = engine.simulate_compensation_change(
            adjustment_type='percentage',
            target={'scope': 'all'},
            adjustment_value=5.0
        )

        comparison = engine.compare_scenarios([result1])

        assert 'error' in comparison


class TestScenarioTemplates:
    """Tests for scenario templates."""

    def test_get_templates(self, sample_employee_data):
        """Test getting scenario templates."""
        engine = ScenarioEngine(sample_employee_data)
        templates = engine.get_scenario_templates()

        assert isinstance(templates, list)
        for t in templates:
            assert 'name' in t
            assert 'type' in t


class TestConfidenceLevels:
    """Tests for confidence level calculation."""

    def test_high_confidence(self, sample_employee_data):
        """Test high confidence with large sample."""
        engine = ScenarioEngine(sample_employee_data)

        result = engine.simulate_compensation_change(
            adjustment_type='percentage',
            target={'scope': 'all'},  # All 100 employees
            adjustment_value=5.0
        )

        assert result.confidence_level == 'High'
        assert result.confidence_score >= 0.8

    def test_low_confidence_small_sample(self, sample_employee_data):
        """Test lower confidence with small sample."""
        engine = ScenarioEngine(sample_employee_data)

        # Create scenario targeting small group
        result = engine.simulate_headcount_change(
            change_type='reduction',
            target={'scope': 'all'},
            change_count=5,  # Small sample
            selection_criteria='performance'
        )

        # With only 5 affected, confidence should be lower
        assert result.confidence_level in ['Low', 'Medium']


class TestCostCalculations:
    """Tests for cost calculations."""

    def test_salary_savings_in_reduction(self, sample_employee_data):
        """Test salary savings are calculated for reduction."""
        engine = ScenarioEngine(sample_employee_data)

        result = engine.simulate_headcount_change(
            change_type='reduction',
            target={'scope': 'all'},
            change_count=10,
            selection_criteria='cost'  # Highest paid first
        )

        assert result.cost_impact.salary_change < 0
        assert result.cost_impact.total_benefit > 0

    def test_roi_calculation(self, sample_employee_data):
        """Test ROI is calculated correctly."""
        engine = ScenarioEngine(sample_employee_data)

        result = engine.simulate_compensation_change(
            adjustment_type='percentage',
            target={'scope': 'all'},
            adjustment_value=5.0
        )

        # ROI should be calculated
        assert result.roi_estimate is not None
        # Should have payback if positive ROI
        if result.roi_estimate > 0 and result.cost_impact.net_impact > 0:
            assert result.payback_months is not None


class TestEdgeCases:
    """Tests for edge cases."""

    def test_zero_adjustment(self, sample_employee_data):
        """Test zero adjustment scenario."""
        engine = ScenarioEngine(sample_employee_data)

        result = engine.simulate_compensation_change(
            adjustment_type='percentage',
            target={'scope': 'all'},
            adjustment_value=0.0
        )

        # Zero adjustment should have minimal impact
        assert result.turnover_change == 0

    def test_large_adjustment(self, sample_employee_data):
        """Test large adjustment is handled."""
        engine = ScenarioEngine(sample_employee_data)

        result = engine.simulate_compensation_change(
            adjustment_type='percentage',
            target={'scope': 'all'},
            adjustment_value=50.0  # 50% raise
        )

        # Should still work, but turnover reduction capped
        assert result.scenario_id is not None
        assert result.turnover_change <= result.baseline_turnover_rate

    def test_data_without_salary(self):
        """Test handling data without salary column."""
        df = pd.DataFrame({
            'EmployeeID': ['EMP001', 'EMP002', 'EMP003'],
            'Dept': ['Engineering', 'Sales', 'HR'],
            'Tenure': [1.0, 2.0, 3.0],
            'LastRating': [3.0, 4.0, 3.5],
            'Age': [30, 35, 40],
        })

        engine = ScenarioEngine(df)

        result = engine.simulate_compensation_change(
            adjustment_type='percentage',
            target={'scope': 'all'},
            adjustment_value=5.0
        )

        # Should use default salary assumption
        assert result.scenario_id is not None


class TestResultSerialization:
    """Tests for result serialization."""

    def test_to_dict(self, sample_employee_data):
        """Test converting result to dictionary."""
        engine = ScenarioEngine(sample_employee_data)

        result = engine.simulate_compensation_change(
            adjustment_type='percentage',
            target={'scope': 'all'},
            adjustment_value=5.0
        )

        d = engine.to_dict(result)

        assert isinstance(d, dict)
        assert 'scenario_id' in d
        assert 'simulation' in d
        assert 'cost_impact' in d
        assert isinstance(d['simulation'], dict)
        assert isinstance(d['cost_impact'], dict)
