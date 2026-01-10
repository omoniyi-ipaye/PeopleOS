"""
Tests for the Preprocessor module.
"""

import pandas as pd
import numpy as np
import pytest

from src.preprocessor import Preprocessor


class TestPreprocessor:
    """Test cases for Preprocessor class."""
    
    def test_impute_missing_values(self, sample_edge_case_data: pd.DataFrame):
        """Test that missing values are imputed correctly."""
        preprocessor = Preprocessor()
        
        # Verify we have nulls to start
        assert sample_edge_case_data['Tenure'].isna().sum() > 0
        
        processed, metadata = preprocessor.fit_transform(sample_edge_case_data)
        
        # After processing, no nulls in numeric columns
        numeric_cols = processed.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            assert processed[col].isna().sum() == 0, f"Column {col} still has nulls"
    
    def test_encode_categorical_variables(self, sample_valid_data: pd.DataFrame):
        """Test that categorical variables are encoded."""
        preprocessor = Preprocessor()
        
        processed, metadata = preprocessor.fit_transform(sample_valid_data)
        
        # Dept should be encoded to numeric
        if 'Dept' in processed.columns:
            assert processed['Dept'].dtype in ['int64', 'int32', 'float64']
    
    def test_scale_numeric_features(self, sample_valid_data: pd.DataFrame):
        """Test that numeric features are scaled."""
        preprocessor = Preprocessor()
        
        processed, metadata = preprocessor.fit_transform(sample_valid_data)
        
        # Salary should be scaled (mean close to 0, std close to 1)
        if 'Salary' in processed.columns and 'Salary' in metadata.get('scaling_columns', []):
            assert abs(processed['Salary'].mean()) < 1
            assert abs(processed['Salary'].std() - 1) < 0.5
    
    def test_cap_outliers(self):
        """Test that outliers are capped using IQR method."""
        # Create data with extreme outliers
        data = pd.DataFrame({
            'EmployeeID': [f'E{i}' for i in range(60)],
            'Dept': ['A'] * 60,
            'Tenure': [5] * 58 + [100, 200],  # Extreme outliers
            'Salary': [50000] * 60,
            'LastRating': [3] * 60,
            'Age': [30] * 60
        })
        
        preprocessor = Preprocessor()
        processed, _ = preprocessor.fit_transform(data)
        
        # Outliers should be capped
        assert processed['Tenure'].max() < 100
    
    def test_preserve_feature_metadata(self, sample_valid_data: pd.DataFrame):
        """Test that feature metadata is preserved."""
        preprocessor = Preprocessor()
        
        processed, metadata = preprocessor.fit_transform(sample_valid_data)
        
        assert 'numeric_columns' in metadata
        assert 'categorical_columns' in metadata
        assert 'processed_columns' in metadata
        assert len(metadata['processed_columns']) > 0
    
    def test_handle_all_null_columns(self):
        """Test handling of columns with all null values."""
        data = pd.DataFrame({
            'EmployeeID': [f'E{i}' for i in range(60)],
            'Dept': ['A'] * 60,
            'Tenure': [5] * 60,
            'Salary': [50000] * 60,
            'LastRating': [3] * 60,
            'Age': [30] * 60,
            'AllNull': [None] * 60
        })
        
        preprocessor = Preprocessor()
        processed, _ = preprocessor.fit_transform(data)
        
        # AllNull column should be dropped (>90% null)
        assert 'AllNull' not in processed.columns
    
    def test_transform_new_data(self, sample_valid_data: pd.DataFrame):
        """Test transforming new data with fitted preprocessor."""
        preprocessor = Preprocessor()
        
        # Fit on original data
        preprocessor.fit_transform(sample_valid_data)
        
        # Transform new similar data
        new_data = sample_valid_data.head(10).copy()
        transformed = preprocessor.transform(new_data)
        
        assert len(transformed) == 10
