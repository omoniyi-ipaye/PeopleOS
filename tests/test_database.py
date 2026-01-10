"""
Tests for the database module.
"""

import os
import tempfile
import pytest
import pandas as pd
from datetime import datetime

from src.database import Database, get_database, reset_database_instance


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
def sample_df():
    """Create sample employee DataFrame."""
    return pd.DataFrame({
        'EmployeeID': ['EMP001', 'EMP002', 'EMP003'],
        'Dept': ['Engineering', 'Sales', 'HR'],
        'Tenure': [3.5, 1.2, 5.0],
        'Salary': [85000, 65000, 72000],
        'LastRating': [4.2, 3.8, 4.5],
        'Age': [32, 28, 35],
        'Attrition': [0, 1, 0],
        'PerformanceText': ['Great leader', 'Needs improvement', 'Excellent team player']
    })


class TestDatabaseInit:
    """Tests for database initialization."""

    def test_creates_database_file(self, temp_db):
        """Test that database file is created."""
        assert os.path.exists(temp_db.db_path)

    def test_creates_tables(self, temp_db):
        """Test that required tables are created."""
        with temp_db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = {row[0] for row in cursor.fetchall()}

        assert 'employees' in tables
        assert 'upload_history' in tables
        assert 'employee_snapshots' in tables

    def test_empty_database_has_no_data(self, temp_db):
        """Test that new database has no employees."""
        assert temp_db.get_employee_count() == 0
        assert not temp_db.has_data()


class TestUpsertEmployees:
    """Tests for upserting employee data."""

    def test_insert_new_employees(self, temp_db, sample_df):
        """Test inserting new employees."""
        result = temp_db.upsert_employees(sample_df, "test_upload.csv")

        assert result['added'] == 3
        assert result['updated'] == 0
        assert result['total'] == 3
        assert temp_db.get_employee_count() == 3

    def test_update_existing_employees(self, temp_db, sample_df):
        """Test updating existing employees."""
        # First insert
        temp_db.upsert_employees(sample_df, "upload1.csv")

        # Modify and re-insert
        modified_df = sample_df.copy()
        modified_df.loc[0, 'Salary'] = 95000  # Raise for EMP001

        result = temp_db.upsert_employees(modified_df, "upload2.csv")

        assert result['updated'] == 3  # All existing
        assert result['added'] == 0
        assert temp_db.get_employee_count() == 3

    def test_mixed_insert_and_update(self, temp_db, sample_df):
        """Test inserting some new and updating some existing."""
        # First insert 2 employees
        temp_db.upsert_employees(sample_df.head(2), "upload1.csv")

        # Now insert all 3 (1 new, 2 updates)
        result = temp_db.upsert_employees(sample_df, "upload2.csv")

        assert result['added'] == 1
        assert result['updated'] == 2
        assert temp_db.get_employee_count() == 3

    def test_skips_invalid_rows(self, temp_db):
        """Test that rows without EmployeeID are skipped."""
        df = pd.DataFrame({
            'EmployeeID': ['EMP001', '', None],
            'Dept': ['Eng', 'Sales', 'HR'],
            'Tenure': [1, 2, 3],
            'Salary': [50000, 60000, 70000],
            'LastRating': [4.0, 3.5, 4.5],
            'Age': [30, 35, 40]
        })

        result = temp_db.upsert_employees(df, "test.csv")

        assert result['added'] == 1
        assert result['skipped'] == 2


class TestGetEmployees:
    """Tests for retrieving employee data."""

    def test_get_all_employees(self, temp_db, sample_df):
        """Test retrieving all employees."""
        temp_db.upsert_employees(sample_df, "test.csv")

        df = temp_db.get_all_employees()

        assert len(df) == 3
        assert 'EmployeeID' in df.columns
        assert 'Dept' in df.columns
        assert set(df['EmployeeID']) == {'EMP001', 'EMP002', 'EMP003'}

    def test_get_employees_excludes_inactive(self, temp_db, sample_df):
        """Test that inactive employees are excluded."""
        temp_db.upsert_employees(sample_df, "test.csv")
        temp_db.soft_delete_employee('EMP002')

        df = temp_db.get_all_employees()

        assert len(df) == 2
        assert 'EMP002' not in df['EmployeeID'].values


class TestHistoricalSnapshots:
    """Tests for historical snapshot functionality."""

    def test_creates_snapshot_on_insert(self, temp_db, sample_df):
        """Test that snapshot is created when employee is inserted."""
        temp_db.upsert_employees(sample_df, "test.csv")

        history = temp_db.get_employee_history('EMP001')

        assert len(history) >= 1
        assert history.iloc[0]['Salary'] == 85000

    def test_creates_snapshot_on_update(self, temp_db, sample_df):
        """Test that snapshot is created when employee is updated."""
        temp_db.upsert_employees(sample_df, "upload1.csv")

        # Update salary
        modified_df = sample_df.copy()
        modified_df.loc[0, 'Salary'] = 95000
        temp_db.upsert_employees(modified_df, "upload2.csv")

        history = temp_db.get_employee_history('EMP001')

        # Should have at least 2 snapshots (initial + before update)
        assert len(history) >= 2

    def test_salary_progression(self, temp_db, sample_df):
        """Test getting salary progression for an employee."""
        temp_db.upsert_employees(sample_df, "upload1.csv")

        progression = temp_db.get_salary_progression('EMP001')

        assert len(progression) >= 1
        assert 'salary' in progression.columns


class TestUploadHistory:
    """Tests for upload history tracking."""

    def test_records_upload_history(self, temp_db, sample_df):
        """Test that uploads are recorded in history."""
        temp_db.upsert_employees(sample_df, "my_upload.csv")

        history = temp_db.get_upload_history()

        assert len(history) >= 1
        assert history[0]['file_name'] == 'my_upload.csv'
        assert history[0]['rows_added'] == 3

    def test_multiple_uploads_tracked(self, temp_db, sample_df):
        """Test that multiple uploads are tracked."""
        temp_db.upsert_employees(sample_df.head(1), "upload1.csv")
        temp_db.upsert_employees(sample_df.tail(2), "upload2.csv")

        history = temp_db.get_upload_history()

        assert len(history) >= 2


class TestSoftDelete:
    """Tests for soft delete functionality."""

    def test_soft_delete_marks_inactive(self, temp_db, sample_df):
        """Test that soft delete marks employee as inactive."""
        temp_db.upsert_employees(sample_df, "test.csv")

        result = temp_db.soft_delete_employee('EMP001')

        assert result is True
        assert temp_db.get_employee_count() == 2

    def test_soft_delete_nonexistent(self, temp_db):
        """Test soft delete of non-existent employee."""
        result = temp_db.soft_delete_employee('NONEXISTENT')
        assert result is False


class TestClearData:
    """Tests for clearing data."""

    def test_clear_all_data(self, temp_db, sample_df):
        """Test clearing all data from database."""
        temp_db.upsert_employees(sample_df, "test.csv")
        assert temp_db.has_data()

        temp_db.clear_all_data()

        assert not temp_db.has_data()
        assert temp_db.get_employee_count() == 0
