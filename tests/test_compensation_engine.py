"""
Tests for the CompensationEngine module.
"""

import pandas as pd
import numpy as np
import pytest

from src.compensation_engine import CompensationEngine, CompensationEngineError


@pytest.fixture
def sample_comp_data():
    """Create sample data for compensation testing."""
    return pd.DataFrame({
        'EmployeeID': [f'E{i:03d}' for i in range(60)],
        'Dept': ['Sales'] * 20 + ['Engineering'] * 20 + ['HR'] * 20,
        'Tenure': np.random.uniform(0.5, 15, 60),
        'Salary': np.concatenate([
            np.random.uniform(50000, 80000, 20),  # Sales
            np.random.uniform(80000, 150000, 20),  # Engineering
            np.random.uniform(45000, 70000, 20)   # HR
        ]),
        'LastRating': np.random.uniform(2, 5, 60),
        'Age': np.random.randint(22, 60, 60),
        'Attrition': np.random.choice([0, 1], 60, p=[0.85, 0.15])
    })


class TestCompensationEngine:
    """Test cases for CompensationEngine class."""

    def test_initialization_success(self, sample_comp_data):
        """Test successful initialization with valid data."""
        engine = CompensationEngine(sample_comp_data)
        assert engine is not None

    def test_calculate_salary_percentiles(self, sample_comp_data):
        """Test salary percentile calculation."""
        engine = CompensationEngine(sample_comp_data)
        percentiles = engine.calculate_salary_percentiles()
        
        assert not percentiles.empty
        assert 'Dept' in percentiles.columns

    def test_calculate_pay_equity_score(self, sample_comp_data):
        """Test pay equity score calculation."""
        engine = CompensationEngine(sample_comp_data)
        equity = engine.calculate_pay_equity_score()
        
        assert not equity.empty
        assert 'Dept' in equity.columns
        assert 'EquityScore' in equity.columns
        # Equity scores should be between 0 and 1
        assert all(0 <= s <= 1 for s in equity['EquityScore'])

    def test_identify_salary_outliers(self, sample_comp_data):
        """Test salary outlier detection."""
        # Add an outlier
        sample_comp_data.loc[0, 'Salary'] = 500000
        
        engine = CompensationEngine(sample_comp_data)
        outliers = engine.identify_salary_outliers()
        
        # Should identify the outlier
        assert len(outliers) >= 1

    def test_get_salary_bands(self, sample_comp_data):
        """Test salary band assignment."""
        engine = CompensationEngine(sample_comp_data)
        bands = engine.get_salary_bands()
        
        assert not bands.empty
        assert 'SalaryBand' in bands.columns

    def test_correlate_salary_with_attrition(self, sample_comp_data):
        """Test salary-attrition correlation."""
        engine = CompensationEngine(sample_comp_data)
        correlation = engine.correlate_salary_with_attrition()
        
        # Should return a dict with correlation data
        assert correlation is not None
        assert isinstance(correlation, dict)
        assert 'correlation' in correlation
        assert -1 <= correlation['correlation'] <= 1

    def test_get_compensation_summary(self, sample_comp_data):
        """Test compensation summary generation."""
        engine = CompensationEngine(sample_comp_data)
        summary = engine.get_compensation_summary()
        
        assert 'avg_salary' in summary
        assert 'median_salary' in summary
        assert summary['avg_salary'] > 0

    def test_analyze_all_returns_dict(self, sample_comp_data):
        """Test full analysis returns complete dictionary."""
        engine = CompensationEngine(sample_comp_data)
        results = engine.analyze_all()
        
        assert isinstance(results, dict)
        assert 'summary' in results
        assert 'equity' in results
