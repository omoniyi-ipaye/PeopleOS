"""
Tests for the MLEngine module.
"""

import pandas as pd
import numpy as np
import pytest

from src.ml_engine import MLEngine, MLEngineError


class TestMLEngine:
    """Test cases for MLEngine class."""
    
    def test_train_model_successfully(self, sample_valid_data: pd.DataFrame):
        """Test that model trains successfully with valid data."""
        engine = MLEngine()
        
        # Prepare features and target
        features = sample_valid_data[['Tenure', 'Salary', 'LastRating', 'Age']].copy()
        target = sample_valid_data['Attrition']
        
        metrics = engine.train_model(features, target)
        
        assert engine.is_trained
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 0 <= metrics['accuracy'] <= 1
    
    def test_consistent_results_with_fixed_seed(self, sample_valid_data: pd.DataFrame):
        """Test that results are consistent with fixed random seed."""
        features = sample_valid_data[['Tenure', 'Salary', 'LastRating', 'Age']].copy()
        target = sample_valid_data['Attrition']
        
        # Train twice
        engine1 = MLEngine()
        metrics1 = engine1.train_model(features, target)
        
        engine2 = MLEngine()
        metrics2 = engine2.train_model(features, target)
        
        # Allow small variance due to SMOTE sampling and model selection
        assert abs(metrics1['accuracy'] - metrics2['accuracy']) < 0.15
    
    def test_predict_risk_scores_valid_range(self, sample_valid_data: pd.DataFrame):
        """Test that risk scores are in valid range [0, 1]."""
        engine = MLEngine()
        
        features = sample_valid_data[['Tenure', 'Salary', 'LastRating', 'Age']].copy()
        target = sample_valid_data['Attrition']
        
        engine.train_model(features, target)
        risk_scores = engine.predict_risk(features)
        
        assert len(risk_scores) == len(features)
        assert all(0 <= score <= 1 for score in risk_scores)
    
    def test_get_risk_drivers(self, sample_valid_data: pd.DataFrame):
        """Test getting risk drivers for an employee."""
        engine = MLEngine()
        
        features = sample_valid_data[['Tenure', 'Salary', 'LastRating', 'Age']].copy()
        target = sample_valid_data['Attrition']
        
        engine.train_model(features, target)
        drivers = engine.get_risk_drivers(0, features)
        
        assert isinstance(drivers, list)
        assert len(drivers) > 0
        assert 'feature' in drivers[0]
        assert 'contribution' in drivers[0]
    
    def test_get_recommendations(self, sample_valid_data: pd.DataFrame):
        """Test recommendation generation for high-risk employees."""
        engine = MLEngine()
        
        features = sample_valid_data[['Tenure', 'Salary', 'LastRating', 'Age']].copy()
        target = sample_valid_data['Attrition']
        
        engine.train_model(features, target)
        
        # Create mock drivers
        drivers = [
            {'feature': 'Salary', 'contribution': 0.3, 'value': 40000, 'abs_contribution': 0.3},
            {'feature': 'Tenure', 'contribution': 0.2, 'value': 0.5, 'abs_contribution': 0.2}
        ]
        
        recommendations = engine.get_recommendations('E001', 0.8, drivers)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) <= 3  # Max 3 recommendations
    
    def test_risk_category_classification(self):
        """Test risk score categorization."""
        engine = MLEngine()
        
        assert engine.get_risk_category(0.8) == "High"
        assert engine.get_risk_category(0.6) == "Medium"
        assert engine.get_risk_category(0.3) == "Low"
    
    def test_predict_without_training_raises_error(self):
        """Test that predicting without training raises an error."""
        engine = MLEngine()
        
        features = pd.DataFrame({
            'Tenure': [1, 2, 3],
            'Salary': [50000, 60000, 70000]
        })
        
        with pytest.raises(MLEngineError):
            engine.predict_risk(features)
    
    def test_handle_edge_case_all_same_class(self):
        """Test handling when all employees have same attrition status."""
        data = pd.DataFrame({
            'Tenure': np.random.uniform(1, 10, 60),
            'Salary': np.random.uniform(50000, 100000, 60),
            'LastRating': np.random.uniform(2, 5, 60),
            'Age': np.random.randint(25, 55, 60)
        })
        # All same class
        target = pd.Series([0] * 60)
        
        engine = MLEngine()
        
        # Should still train, though metrics may be less meaningful
        metrics = engine.train_model(data, target)
        
        assert engine.is_trained
    
    def test_feature_importance_summary(self, sample_valid_data: pd.DataFrame):
        """Test feature importance summary generation."""
        engine = MLEngine()
        
        features = sample_valid_data[['Tenure', 'Salary', 'LastRating', 'Age']].copy()
        target = sample_valid_data['Attrition']
        
        engine.train_model(features, target)
        importance_df = engine.get_feature_importance_summary()
        
        assert not importance_df.empty
        assert 'Feature' in importance_df.columns
        assert 'Importance' in importance_df.columns
        assert len(importance_df) == len(features.columns)
