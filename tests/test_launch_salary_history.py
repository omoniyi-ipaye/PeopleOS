"""Known-answer salary observations must retain their actual dates and values."""

import pandas as pd
import pytest

from src.database import Database


@pytest.fixture
def database(tmp_path):
    db = Database(str(tmp_path / 'history.sqlite'))
    db.keep_history = True
    return db


def observation(salary, date=None):
    row = {'EmployeeID': 'E001', 'Dept': 'Engineering', 'Salary': salary,
           'Tenure': 3, 'LastRating': 4, 'Age': 35, 'Attrition': 0}
    if date is not None:
        row['SnapshotDate'] = date
    return pd.DataFrame([row])


def test_salary_observation_dates_are_not_shifted(database):
    for year, salary in [(2024, 50000), (2025, 60000), (2026, 70000)]:
        result = database.upsert_employees(observation(salary, f'{year}-01-01'))
        assert result['total'] == 1
        assert result['skipped'] == 0
    history = database.get_employee_history('E001')
    assert pd.to_datetime(history.snapshot_date).dt.year.tolist() == [2024, 2025, 2026]
    assert history.Salary.tolist() == [50000, 60000, 70000]
    assert database.get_salary_progression('E001').salary.tolist() == [50000, 60000, 70000]
    assert database.get_all_employees().Salary.tolist() == [70000]
    assert database.get_historical_snapshots('2025-01-01').Salary.tolist() == [60000, 70000]


def test_equal_dates_retain_observation_order(database):
    for salary in [50000, 60000, 70000]:
        database.upsert_employees(observation(salary, pd.Timestamp('2026-01-01', tz='UTC')))
    assert database.get_employee_history('E001').Salary.tolist() == [50000, 60000, 70000]
    assert database.get_historical_snapshots('2025-01-01').Salary.tolist() == [50000, 60000, 70000]


def test_undated_uploads_include_current_state(database):
    for salary in [50000, 60000, 70000]:
        database.upsert_employees(observation(salary))
    history = database.get_employee_history('E001')
    assert history.Salary.tolist() == [50000, 60000, 70000]
    assert history.snapshot_date.notna().all()


def test_timestamp_offsets_sort_by_actual_observation_time(database):
    database.upsert_employees(observation(50000, '2026-01-01T02:00:00+02:00'))
    database.upsert_employees(observation(60000, '2026-01-01T01:00:00Z'))
    assert database.get_employee_history('E001').Salary.tolist() == [50000, 60000]


def test_invalid_date_does_not_partially_overwrite_employee(database):
    database.upsert_employees(observation(50000, '2024-01-01'))
    result = database.upsert_employees(observation(70000, 'not-a-date'))
    assert result == {'added': 0, 'updated': 0, 'skipped': 1, 'total': 0}
    assert database.get_all_employees().Salary.tolist() == [50000]
    assert database.get_employee_history('E001').Salary.tolist() == [50000]


@pytest.mark.parametrize('existing', [False, True])
def test_snapshot_write_failure_rolls_back_the_employee(database, monkeypatch, existing):
    if existing:
        database.upsert_employees(observation(50000, '2024-01-01'))

    def fail_snapshot(*args, **kwargs):
        raise RuntimeError('simulated snapshot write failure')

    monkeypatch.setattr(database, '_create_snapshot', fail_snapshot)
    result = database.upsert_employees(observation(70000, '2026-01-01'))
    assert result == {'added': 0, 'updated': 0, 'skipped': 1, 'total': 0}
    current = database.get_all_employees()
    history = database.get_employee_history('E001')
    assert current.Salary.tolist() == ([50000] if existing else [])
    assert history.Salary.tolist() == ([50000] if existing else [])
