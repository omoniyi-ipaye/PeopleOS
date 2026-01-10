"""
Tests for the DataLoader module.
"""

import os
import json
import tempfile

import pandas as pd
import pytest

from src.data_loader import DataLoader, DataValidationError


class TestDataLoader:
    """Test cases for DataLoader class."""
    
    def test_load_valid_csv(self, sample_valid_data: pd.DataFrame):
        """Test loading a valid CSV file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_valid_data.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            loader = DataLoader()
            df = loader.load(temp_path)
            
            assert len(df) == len(sample_valid_data)
            assert 'EmployeeID' in df.columns
            assert 'Dept' in df.columns
        finally:
            os.unlink(temp_path)
    
    def test_load_valid_json(self, sample_valid_data: pd.DataFrame):
        """Test loading a valid JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            sample_valid_data.to_json(f.name, orient='records')
            temp_path = f.name
        
        try:
            loader = DataLoader()
            df = loader.load(temp_path)
            
            assert len(df) == len(sample_valid_data)
        finally:
            os.unlink(temp_path)
    
    def test_reject_missing_required_columns(self, sample_invalid_data: pd.DataFrame):
        """Test that files with missing required columns are rejected."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_invalid_data.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            loader = DataLoader()
            with pytest.raises(DataValidationError) as exc_info:
                loader.load(temp_path)
            
            assert "missing" in str(exc_info.value).lower()
        finally:
            os.unlink(temp_path)
    
    def test_fuzzy_column_matching(self, sample_data_with_fuzzy_columns: pd.DataFrame):
        """Test that fuzzy column name matching works."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_data_with_fuzzy_columns.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            loader = DataLoader()
            df = loader.load(temp_path)
            
            # Check that columns were mapped to Golden Schema
            assert 'EmployeeID' in df.columns
            assert 'Dept' in df.columns
            assert 'Tenure' in df.columns
            assert 'Salary' in df.columns
            
            # Check mapping report
            report = loader.get_column_mapping_report()
            assert len(report['mappings']) > 0
        finally:
            os.unlink(temp_path)
    
    def test_reject_duplicate_employee_ids(self, sample_data_with_duplicates: pd.DataFrame):
        """Test that files with duplicate EmployeeIDs are rejected."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_data_with_duplicates.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            loader = DataLoader()
            with pytest.raises(DataValidationError) as exc_info:
                loader.load(temp_path)
            
            # HR-friendly error message mentions "same ID" instead of "duplicate"
            assert "same id" in str(exc_info.value).lower() or "duplicate" in str(exc_info.value).lower()
        finally:
            os.unlink(temp_path)
    
    def test_reject_insufficient_rows(self, sample_data_too_few_rows: pd.DataFrame):
        """Test that files with insufficient rows are rejected."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_data_too_few_rows.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            loader = DataLoader()
            with pytest.raises(DataValidationError) as exc_info:
                loader.load(temp_path)
            
            assert "50" in str(exc_info.value) or "rows" in str(exc_info.value).lower()
        finally:
            os.unlink(temp_path)
    
    def test_handle_edge_case_data(self, sample_edge_case_data: pd.DataFrame):
        """Test handling of edge case data with nulls."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_edge_case_data.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            loader = DataLoader()
            df = loader.load(temp_path)
            
            # Should succeed even with some nulls
            assert len(df) > 0
        finally:
            os.unlink(temp_path)
    
    def test_features_disabled_without_attrition(self, sample_valid_data: pd.DataFrame):
        """Test that predictive features are disabled without Attrition column."""
        # Remove Attrition column
        data_no_attrition = sample_valid_data.drop(columns=['Attrition'])
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            data_no_attrition.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            loader = DataLoader()
            loader.load(temp_path)
            
            report = loader.get_column_mapping_report()
            assert report['features_enabled']['predictive'] is False
        finally:
            os.unlink(temp_path)
