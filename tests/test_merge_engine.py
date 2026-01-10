"""
Tests for the merge engine module.
"""

import os
import tempfile
import pytest
import pandas as pd

from src.database import Database, reset_database_instance
from src.merge_engine import MergeEngine, MergeResult, FieldChange, EmployeeChange


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name

    db = Database(db_path=db_path)
    yield db

    # Cleanup
    reset_database_instance()
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
def merge_engine(temp_db):
    """Create merge engine with temp database."""
    return MergeEngine(database=temp_db)


@pytest.fixture
def sample_df():
    """Create sample employee DataFrame."""
    return pd.DataFrame({
        'EmployeeID': ['EMP001', 'EMP002', 'EMP003'],
        'Dept': ['Engineering', 'Sales', 'HR'],
        'Tenure': [3.5, 1.2, 5.0],
        'Salary': [85000, 65000, 72000],
        'LastRating': [4.2, 3.8, 4.5],
        'Age': [32, 28, 35],
        'Attrition': [0, 1, 0]
    })


class TestMergeResult:
    """Tests for MergeResult dataclass."""

    def test_summary_text_all_new(self):
        """Test summary text when all employees are new."""
        result = MergeResult(added=5, updated=0, unchanged=0, total=5)
        summary = result.get_summary_text()

        assert "5 new employees added" in summary

    def test_summary_text_all_updated(self):
        """Test summary text when all employees are updated."""
        result = MergeResult(added=0, updated=3, unchanged=0, total=3)
        summary = result.get_summary_text()

        assert "3 employees updated" in summary

    def test_summary_text_mixed(self):
        """Test summary text with mixed changes."""
        result = MergeResult(added=2, updated=3, unchanged=1, skipped=1, total=6)
        summary = result.get_summary_text()

        assert "2 new" in summary
        assert "3 employee" in summary and "updated" in summary

    def test_summary_text_no_changes(self):
        """Test summary text when no changes made."""
        result = MergeResult()
        summary = result.get_summary_text()

        assert summary == "No changes made"


class TestFieldChange:
    """Tests for FieldChange dataclass."""

    def test_to_dict(self):
        """Test converting FieldChange to dictionary."""
        change = FieldChange(field_name='Salary', old_value=50000, new_value=60000)
        d = change.to_dict()

        assert d['field'] == 'Salary'
        assert d['old'] == 50000
        assert d['new'] == 60000


class TestPreviewMerge:
    """Tests for merge preview functionality."""

    def test_preview_all_new(self, merge_engine, sample_df):
        """Test preview when all employees are new."""
        result = merge_engine.preview_merge(sample_df)

        assert result.added == 3
        assert result.updated == 0
        assert result.unchanged == 0

    def test_preview_all_unchanged(self, merge_engine, temp_db, sample_df):
        """Test preview when no changes."""
        # First insert
        temp_db.upsert_employees(sample_df, "initial.csv")

        # Preview same data
        result = merge_engine.preview_merge(sample_df)

        assert result.added == 0
        assert result.updated == 0
        assert result.unchanged == 3

    def test_preview_detects_changes(self, merge_engine, temp_db, sample_df):
        """Test preview detects field changes."""
        temp_db.upsert_employees(sample_df, "initial.csv")

        # Modify data
        modified_df = sample_df.copy()
        modified_df.loc[0, 'Salary'] = 95000  # Change salary for EMP001

        result = merge_engine.preview_merge(modified_df)

        assert result.updated == 1
        assert result.unchanged == 2

        # Find the updated employee
        updated_emp = [e for e in result.employee_changes if e.change_type == 'updated'][0]
        assert updated_emp.employee_id == 'EMP001'
        assert len(updated_emp.changes) == 1
        assert updated_emp.changes[0].field_name == 'Salary'

    def test_preview_mixed_scenario(self, merge_engine, temp_db, sample_df):
        """Test preview with mix of new, updated, unchanged."""
        # Insert first 2 employees
        temp_db.upsert_employees(sample_df.head(2), "initial.csv")

        # Create new data: EMP001 unchanged, EMP002 updated, EMP003 new
        new_df = sample_df.copy()
        new_df.loc[1, 'Tenure'] = 2.0  # Update EMP002's tenure

        result = merge_engine.preview_merge(new_df)

        assert result.added == 1  # EMP003
        assert result.updated == 1  # EMP002
        assert result.unchanged == 1  # EMP001


class TestExecuteMerge:
    """Tests for executing merges."""

    def test_execute_merge_inserts(self, merge_engine, temp_db, sample_df):
        """Test that execute_merge actually inserts data."""
        result = merge_engine.execute_merge(sample_df, "test_upload.csv")

        assert result.added == 3
        assert temp_db.get_employee_count() == 3

    def test_execute_merge_updates(self, merge_engine, temp_db, sample_df):
        """Test that execute_merge updates existing records."""
        merge_engine.execute_merge(sample_df, "upload1.csv")

        modified_df = sample_df.copy()
        modified_df.loc[0, 'Salary'] = 100000

        result = merge_engine.execute_merge(modified_df, "upload2.csv")

        assert result.updated >= 1

        # Verify the update persisted
        employees = temp_db.get_all_employees()
        emp001 = employees[employees['EmployeeID'] == 'EMP001'].iloc[0]
        assert emp001['Salary'] == 100000

    def test_execute_merge_records_file_name(self, merge_engine, temp_db, sample_df):
        """Test that file name is recorded in result."""
        result = merge_engine.execute_merge(sample_df, "my_data.csv")

        assert result.file_name == "my_data.csv"


class TestSignificantChanges:
    """Tests for detecting significant changes."""

    def test_detects_large_salary_change(self, merge_engine, temp_db, sample_df):
        """Test detection of large salary changes (>10%)."""
        temp_db.upsert_employees(sample_df, "initial.csv")

        # 20% salary increase for EMP001
        modified_df = sample_df.copy()
        modified_df.loc[0, 'Salary'] = 102000  # 85000 * 1.2

        result = merge_engine.preview_merge(modified_df)
        significant = merge_engine.get_significant_changes(result, salary_threshold=0.10)

        assert len(significant) >= 1
        assert any(e.employee_id == 'EMP001' for e in significant)

    def test_detects_department_change(self, merge_engine, temp_db, sample_df):
        """Test detection of department transfers."""
        temp_db.upsert_employees(sample_df, "initial.csv")

        modified_df = sample_df.copy()
        modified_df.loc[0, 'Dept'] = 'Marketing'  # Transfer EMP001

        result = merge_engine.preview_merge(modified_df)
        significant = merge_engine.get_significant_changes(result)

        assert len(significant) >= 1

    def test_detects_attrition_change(self, merge_engine, temp_db, sample_df):
        """Test detection of attrition status changes."""
        temp_db.upsert_employees(sample_df, "initial.csv")

        modified_df = sample_df.copy()
        modified_df.loc[0, 'Attrition'] = 1  # EMP001 left

        result = merge_engine.preview_merge(modified_df)
        significant = merge_engine.get_significant_changes(result)

        assert len(significant) >= 1

    def test_ignores_small_changes(self, merge_engine, temp_db, sample_df):
        """Test that small changes are not flagged."""
        temp_db.upsert_employees(sample_df, "initial.csv")

        # Small 2% salary change
        modified_df = sample_df.copy()
        modified_df.loc[0, 'Salary'] = 86700  # ~2% increase

        result = merge_engine.preview_merge(modified_df)
        significant = merge_engine.get_significant_changes(result, salary_threshold=0.10)

        # Should not be flagged as significant
        assert not any(e.employee_id == 'EMP001' for e in significant)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_dataframe(self, merge_engine):
        """Test handling empty DataFrame."""
        df = pd.DataFrame()
        result = merge_engine.preview_merge(df)

        assert result.total == 0

    def test_missing_employee_id(self, merge_engine):
        """Test handling rows without EmployeeID."""
        df = pd.DataFrame({
            'EmployeeID': ['EMP001', '', None],
            'Dept': ['Eng', 'Sales', 'HR'],
            'Tenure': [1, 2, 3],
            'Salary': [50000, 60000, 70000],
            'LastRating': [4.0, 3.5, 4.5],
            'Age': [30, 35, 40]
        })

        result = merge_engine.preview_merge(df)

        assert result.skipped == 2
        assert result.added == 1

    def test_null_value_handling(self, merge_engine, temp_db):
        """Test handling of null values in comparisons."""
        # Insert with null performance text
        df1 = pd.DataFrame({
            'EmployeeID': ['EMP001'],
            'Dept': ['Engineering'],
            'Tenure': [3.5],
            'Salary': [85000],
            'LastRating': [4.2],
            'Age': [32],
            'Attrition': [None]
        })
        temp_db.upsert_employees(df1, "initial.csv")

        # Update with same null value
        result = merge_engine.preview_merge(df1)

        # Should be unchanged, not flagged as updated
        assert result.unchanged == 1
