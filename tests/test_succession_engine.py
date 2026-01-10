"""
Tests for the SuccessionEngine module.
"""

import pandas as pd
import numpy as np
import pytest

from src.succession_engine import SuccessionEngine, SuccessionEngineError


@pytest.fixture
def sample_succ_data():
    """Create sample data for succession testing."""
    return pd.DataFrame({
        'EmployeeID': [f'E{i:03d}' for i in range(60)],
        'Dept': ['Sales'] * 20 + ['Engineering'] * 20 + ['HR'] * 20,
        'Tenure': np.random.uniform(0.5, 15, 60),
        'Salary': np.random.uniform(50000, 150000, 60),
        'LastRating': np.random.uniform(2, 5, 60),
        'Age': np.random.randint(22, 60, 60)
    })


@pytest.fixture
def sample_risk_scores():
    """Create sample risk scores DataFrame."""
    return pd.DataFrame({
        'EmployeeID': [f'E{i:03d}' for i in range(60)],
        'risk_score': np.random.uniform(0, 1, 60)
    })


class TestSuccessionEngine:
    """Test cases for SuccessionEngine class."""

    def test_initialization_success(self, sample_succ_data):
        """Test successful initialization with valid data."""
        engine = SuccessionEngine(sample_succ_data)
        assert engine is not None

    def test_initialization_with_risk_scores(self, sample_succ_data, sample_risk_scores):
        """Test initialization with risk scores."""
        engine = SuccessionEngine(sample_succ_data, sample_risk_scores)
        assert engine.risk_scores is not None

    def test_calculate_readiness_scores(self, sample_succ_data):
        """Test readiness score calculation."""
        engine = SuccessionEngine(sample_succ_data)
        readiness = engine.calculate_readiness_scores()
        
        assert not readiness.empty
        assert 'ReadinessScore' in readiness.columns
        assert 'ReadinessLevel' in readiness.columns
        # Scores should be between 0 and 1
        assert all(0 <= s <= 1 for s in readiness['ReadinessScore'])

    def test_identify_high_potentials(self, sample_succ_data):
        """Test high-potential identification."""
        # Ensure some high performers exist
        sample_succ_data.loc[:10, 'LastRating'] = 4.5
        sample_succ_data.loc[:10, 'Tenure'] = 3.0
        
        engine = SuccessionEngine(sample_succ_data)
        hi_pos = engine.identify_high_potentials()
        
        # Should identify some high potentials
        assert len(hi_pos) > 0
        assert 'PotentialLevel' in hi_pos.columns

    def test_get_succession_pipeline(self, sample_succ_data):
        """Test succession pipeline generation."""
        engine = SuccessionEngine(sample_succ_data)
        pipeline = engine.get_succession_pipeline()
        
        assert isinstance(pipeline, dict)
        # Should have entries for each department
        assert len(pipeline) > 0

    def test_calculate_bench_strength(self, sample_succ_data):
        """Test bench strength calculation."""
        engine = SuccessionEngine(sample_succ_data)
        bench = engine.calculate_bench_strength()
        
        assert not bench.empty
        assert 'Dept' in bench.columns
        assert 'BenchStrength' in bench.columns

    def test_get_9box_matrix(self, sample_succ_data):
        """Test 9-box matrix generation."""
        engine = SuccessionEngine(sample_succ_data)
        matrix = engine.get_9box_matrix()
        
        assert not matrix.empty
        assert 'NineBox' in matrix.columns

    def test_get_9box_summary(self, sample_succ_data):
        """Test 9-box summary counts."""
        engine = SuccessionEngine(sample_succ_data)
        summary = engine.get_9box_summary()
        
        assert not summary.empty
        # Total should equal employee count
        assert summary['Count'].sum() == len(sample_succ_data)

    def test_analyze_all_returns_dict(self, sample_succ_data):
        """Test full analysis returns complete dictionary."""
        engine = SuccessionEngine(sample_succ_data)
        results = engine.analyze_all()
        
        assert isinstance(results, dict)
        assert 'readiness' in results
        assert 'high_potentials' in results
        assert 'bench_strength' in results
