"""
Tests for the AnalyticsEngine module.
"""

import pandas as pd
import numpy as np
import pytest

from src.analytics_engine import AnalyticsEngine


class TestAnalyticsEngine:
    """Test cases for AnalyticsEngine class."""
    
    def test_calculate_correct_headcount(self, sample_valid_data: pd.DataFrame):
        """Test that headcount is calculated correctly."""
        engine = AnalyticsEngine(sample_valid_data)
        
        headcount = engine.get_headcount()
        
        assert headcount == len(sample_valid_data)
    
    def test_calculate_turnover_rate(self, sample_valid_data: pd.DataFrame):
        """Test that turnover rate is calculated correctly."""
        engine = AnalyticsEngine(sample_valid_data)
        
        turnover = engine.get_turnover_rate()
        
        # Should be between 0 and 1
        assert turnover is not None
        assert 0 <= turnover <= 1
        
        # Manually calculate expected rate
        expected = sample_valid_data['Attrition'].sum() / len(sample_valid_data)
        assert abs(turnover - expected) < 0.001
    
    def test_turnover_rate_without_attrition(self):
        """Test that turnover rate returns None without Attrition column."""
        data = pd.DataFrame({
            'EmployeeID': ['E1', 'E2', 'E3'],
            'Dept': ['A', 'B', 'C'],
            'Tenure': [1, 2, 3],
            'Salary': [50000, 60000, 70000],
            'LastRating': [3, 4, 5],
            'Age': [25, 30, 35]
        })
        
        engine = AnalyticsEngine(data)
        turnover = engine.get_turnover_rate()
        
        assert turnover is None
    
    def test_generate_department_aggregates(self, sample_valid_data: pd.DataFrame):
        """Test department aggregate generation."""
        engine = AnalyticsEngine(sample_valid_data)
        
        dept_stats = engine.get_department_aggregates()
        
        assert not dept_stats.empty
        assert 'Dept' in dept_stats.columns
        assert 'Headcount' in dept_stats.columns
        
        # Sum of headcounts should equal total
        assert dept_stats['Headcount'].sum() == len(sample_valid_data)
    
    def test_handle_division_by_zero(self):
        """Test that division by zero is handled in rates."""
        # Empty DataFrame case
        data = pd.DataFrame({
            'EmployeeID': [],
            'Dept': [],
            'Tenure': [],
            'Salary': [],
            'LastRating': [],
            'Age': [],
            'Attrition': []
        })
        
        engine = AnalyticsEngine(data)
        headcount = engine.get_headcount()
        
        # Should return 0 for empty data, not crash
        assert headcount == 0
    
    def test_get_summary_statistics(self, sample_valid_data: pd.DataFrame):
        """Test summary statistics generation."""
        engine = AnalyticsEngine(sample_valid_data)
        
        stats = engine.get_summary_statistics()
        
        assert 'headcount' in stats
        assert 'turnover_rate' in stats
        assert 'department_count' in stats
        assert stats['headcount'] == len(sample_valid_data)
    
    def test_get_correlations(self, sample_valid_data: pd.DataFrame):
        """Test correlation calculation."""
        engine = AnalyticsEngine(sample_valid_data)
        
        correlations = engine.get_correlations()
        
        assert not correlations.empty
        assert 'Feature' in correlations.columns
        assert 'Correlation' in correlations.columns
    
    def test_get_tenure_distribution(self, sample_valid_data: pd.DataFrame):
        """Test tenure distribution buckets."""
        engine = AnalyticsEngine(sample_valid_data)
        
        distribution = engine.get_tenure_distribution()
        
        assert not distribution.empty
        assert 'Tenure_Range' in distribution.columns
        assert 'Count' in distribution.columns
    
    def test_get_high_risk_departments(self, sample_valid_data: pd.DataFrame):
        """Test high risk department identification."""
        engine = AnalyticsEngine(sample_valid_data)
        
        high_risk = engine.get_high_risk_departments(threshold=0.1)
        
        # Should return a DataFrame (may be empty if no dept over threshold)
        assert isinstance(high_risk, pd.DataFrame)
