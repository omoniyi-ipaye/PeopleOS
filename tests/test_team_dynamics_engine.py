"""
Tests for the TeamDynamicsEngine module.
"""

import pandas as pd
import numpy as np
import pytest

from src.team_dynamics_engine import TeamDynamicsEngine, TeamDynamicsEngineError


@pytest.fixture
def sample_team_data():
    """Create sample data for team dynamics testing."""
    return pd.DataFrame({
        'EmployeeID': [f'E{i:03d}' for i in range(60)],
        'Dept': ['Sales'] * 20 + ['Engineering'] * 20 + ['HR'] * 20,
        'Tenure': np.random.uniform(0.5, 15, 60),
        'Salary': np.random.uniform(50000, 150000, 60),
        'LastRating': np.random.uniform(2, 5, 60),
        'Age': np.random.randint(22, 60, 60),
        'Attrition': np.random.choice([0, 1], 60, p=[0.85, 0.15])
    })


@pytest.fixture
def sample_nlp_data():
    """Create sample NLP data for sentiment integration."""
    return {
        'sentiment': pd.DataFrame({
            'EmployeeID': [f'E{i:03d}' for i in range(60)],
            'sentiment_score': np.random.uniform(0, 1, 60),
            'sentiment_label': np.random.choice(['Positive', 'Neutral', 'Negative'], 60)
        })
    }


class TestTeamDynamicsEngine:
    """Test cases for TeamDynamicsEngine class."""

    def test_initialization_success(self, sample_team_data):
        """Test successful initialization with valid data."""
        engine = TeamDynamicsEngine(sample_team_data)
        assert engine is not None

    def test_initialization_with_nlp_data(self, sample_team_data, sample_nlp_data):
        """Test initialization with NLP data."""
        engine = TeamDynamicsEngine(sample_team_data, sample_nlp_data)
        assert engine.nlp_data is not None

    def test_calculate_team_health_scores(self, sample_team_data):
        """Test team health score calculation."""
        engine = TeamDynamicsEngine(sample_team_data)
        health = engine.calculate_team_health_scores()
        
        assert not health.empty
        assert 'Dept' in health.columns
        assert 'HealthScore' in health.columns
        # Health scores should be between 0 and 1
        assert all(0 <= s <= 1 for s in health['HealthScore'])

    def test_analyze_team_diversity(self, sample_team_data):
        """Test team diversity analysis."""
        engine = TeamDynamicsEngine(sample_team_data)
        diversity = engine.analyze_team_diversity()
        
        assert not diversity.empty
        assert 'Dept' in diversity.columns

    def test_identify_performance_variance(self, sample_team_data):
        """Test performance variance identification."""
        engine = TeamDynamicsEngine(sample_team_data)
        variance = engine.identify_performance_variance()
        
        assert not variance.empty
        assert 'Dept' in variance.columns
        assert 'RatingStdDev' in variance.columns

    def test_get_collaboration_indicators(self, sample_team_data):
        """Test collaboration indicators."""
        engine = TeamDynamicsEngine(sample_team_data)
        collab = engine.get_collaboration_indicators()
        
        assert isinstance(collab, dict)
        assert 'department_count' in collab

    def test_flag_at_risk_teams(self, sample_team_data):
        """Test at-risk team flagging."""
        engine = TeamDynamicsEngine(sample_team_data)
        at_risk = engine.flag_at_risk_teams()
        
        # May be empty if no teams are at risk
        assert isinstance(at_risk, pd.DataFrame)

    def test_get_team_composition(self, sample_team_data):
        """Test team composition analysis."""
        engine = TeamDynamicsEngine(sample_team_data)
        composition = engine.get_team_composition()
        
        assert not composition.empty
        assert 'Dept' in composition.columns

    def test_get_team_summary(self, sample_team_data):
        """Test team summary generation."""
        engine = TeamDynamicsEngine(sample_team_data)
        summary = engine.get_team_summary()
        
        assert isinstance(summary, dict)
        assert 'total_teams' in summary

    def test_analyze_all_returns_dict(self, sample_team_data):
        """Test full analysis returns complete dictionary."""
        engine = TeamDynamicsEngine(sample_team_data)
        results = engine.analyze_all()
        
        assert isinstance(results, dict)
        assert 'health' in results
        assert 'diversity' in results
        assert 'composition' in results
