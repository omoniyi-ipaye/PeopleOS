"""
Shared test fixtures for PeopleOS tests.
"""

import pandas as pd
import pytest
import numpy as np


@pytest.fixture
def sample_valid_data() -> pd.DataFrame:
    """Returns a valid sample DataFrame for testing."""
    np.random.seed(42)
    n = 100
    
    return pd.DataFrame({
        'EmployeeID': [f'EMP{i:04d}' for i in range(1, n + 1)],
        'Dept': np.random.choice(['Engineering', 'Sales', 'HR', 'Marketing', 'Finance'], n),
        'Tenure': np.random.uniform(0.5, 15, n).round(1),
        'Salary': np.random.uniform(40000, 150000, n).round(0),
        'LastRating': np.random.uniform(1, 5, n).round(1),
        'Age': np.random.randint(22, 65, n),
        'Attrition': np.random.choice([0, 1], n, p=[0.85, 0.15])
    })


@pytest.fixture
def sample_invalid_data() -> pd.DataFrame:
    """Returns an invalid DataFrame (missing columns, bad types)."""
    n = 60
    return pd.DataFrame({
        'ID': list(range(1, n + 1)),
        'Department': ['A', 'B', 'C'] * 20,
        'Years': list(range(1, n + 1))
    })


@pytest.fixture
def sample_edge_case_data() -> pd.DataFrame:
    """Returns edge case data (min rows, nulls, duplicates)."""
    np.random.seed(42)
    n = 55  # Just above minimum
    
    data = pd.DataFrame({
        'EmployeeID': [f'EMP{i:04d}' for i in range(1, n + 1)],
        'Dept': np.random.choice(['Engineering', 'Sales'], n),
        'Tenure': np.random.uniform(0.5, 10, n).round(1),
        'Salary': np.random.uniform(50000, 100000, n).round(0),
        'LastRating': np.random.uniform(2, 5, n).round(1),
        'Age': np.random.randint(25, 55, n),
        'Attrition': np.random.choice([0, 1], n)
    })
    
    # Add some nulls
    data.loc[0, 'Tenure'] = None
    data.loc[1, 'Salary'] = None
    
    return data


@pytest.fixture
def sample_data_with_fuzzy_columns() -> pd.DataFrame:
    """Returns data with fuzzy column names."""
    np.random.seed(42)
    n = 60
    
    return pd.DataFrame({
        'emp_id': [f'E{i}' for i in range(1, n + 1)],
        'department': np.random.choice(['Eng', 'Sales'], n),
        'years_of_service': np.random.uniform(1, 10, n).round(1),
        'compensation': np.random.uniform(50000, 100000, n).round(0),
        'rating': np.random.uniform(2, 5, n).round(1),
        'employee_age': np.random.randint(25, 55, n)
    })


@pytest.fixture
def sample_data_with_duplicates() -> pd.DataFrame:
    """Returns data with duplicate EmployeeIDs."""
    np.random.seed(42)
    n = 60
    
    data = pd.DataFrame({
        'EmployeeID': [f'EMP{i:04d}' for i in range(1, n + 1)],
        'Dept': np.random.choice(['Engineering', 'Sales'], n),
        'Tenure': np.random.uniform(1, 10, n).round(1),
        'Salary': np.random.uniform(50000, 100000, n).round(0),
        'LastRating': np.random.uniform(2, 5, n).round(1),
        'Age': np.random.randint(25, 55, n)
    })
    
    # Add duplicates
    data.loc[5, 'EmployeeID'] = data.loc[0, 'EmployeeID']
    data.loc[10, 'EmployeeID'] = data.loc[0, 'EmployeeID']
    
    return data


@pytest.fixture
def sample_data_too_few_rows() -> pd.DataFrame:
    """Returns data with too few rows."""
    return pd.DataFrame({
        'EmployeeID': ['E1', 'E2', 'E3'],
        'Dept': ['A', 'B', 'C'],
        'Tenure': [1, 2, 3],
        'Salary': [50000, 60000, 70000],
        'LastRating': [3, 4, 5],
        'Age': [25, 30, 35]
    })


@pytest.fixture
def sample_data_minimum_rows() -> pd.DataFrame:
    """Returns data with exactly 50 rows (minimum allowed)."""
    np.random.seed(42)
    n = 50

    return pd.DataFrame({
        'EmployeeID': [f'EMP{i:04d}' for i in range(1, n + 1)],
        'Dept': np.random.choice(['Engineering', 'Sales', 'HR'], n),
        'Tenure': np.random.uniform(0.5, 10, n).round(1),
        'Salary': np.random.uniform(40000, 100000, n).round(0),
        'LastRating': np.random.uniform(1, 5, n).round(1),
        'Age': np.random.randint(22, 60, n),
        'Attrition': np.random.choice([0, 1], n, p=[0.8, 0.2])
    })


@pytest.fixture
def sample_data_large() -> pd.DataFrame:
    """Returns larger dataset for performance testing (1000 rows)."""
    np.random.seed(42)
    n = 1000

    return pd.DataFrame({
        'EmployeeID': [f'EMP{i:05d}' for i in range(1, n + 1)],
        'Dept': np.random.choice(['Engineering', 'Sales', 'HR', 'Marketing', 'Finance', 'Operations'], n),
        'Tenure': np.random.uniform(0.1, 20, n).round(1),
        'Salary': np.random.uniform(30000, 200000, n).round(0),
        'LastRating': np.random.uniform(1, 5, n).round(1),
        'Age': np.random.randint(20, 65, n),
        'Attrition': np.random.choice([0, 1], n, p=[0.85, 0.15])
    })


@pytest.fixture
def sample_data_single_class() -> pd.DataFrame:
    """Returns data with single class attrition (all zeros)."""
    np.random.seed(42)
    n = 100

    return pd.DataFrame({
        'EmployeeID': [f'EMP{i:04d}' for i in range(1, n + 1)],
        'Dept': np.random.choice(['Engineering', 'Sales', 'HR'], n),
        'Tenure': np.random.uniform(0.5, 10, n).round(1),
        'Salary': np.random.uniform(40000, 100000, n).round(0),
        'LastRating': np.random.uniform(1, 5, n).round(1),
        'Age': np.random.randint(22, 60, n),
        'Attrition': np.zeros(n, dtype=int)  # All zeros - no attrition
    })
