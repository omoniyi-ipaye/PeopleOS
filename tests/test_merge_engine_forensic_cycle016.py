"""Cycle 016 forensic contracts for MergeEngine."""
from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.database import Database, reset_database_instance
from src.merge_engine import MergeEngine


@pytest.fixture
def db():
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as handle:
        path = handle.name
    database = Database(db_path=path)
    yield database
    reset_database_instance()
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def engine(db):
    return MergeEngine(database=db)


def workforce() -> pd.DataFrame:
    return pd.DataFrame({
        'EmployeeID': ['E001', 'E002'],
        'Dept': ['People', 'Engineering'],
        'Tenure': [3.0, 4.0],
        'Salary': [70000.0, 90000.0],
        'LastRating': [4.0, 4.5],
        'Age': [35, 38],
        'Attrition': [0, 0],
        'Gender': ['Female', 'Male'],
        'JobTitle': ['Partner', 'Engineer'],
        'Location': ['Madrid', 'London'],
        'ManagerID': ['M001', 'M002'],
    })


def test_unknown_sensitive_columns_never_enter_change_preview(engine, db):
    frame = workforce()
    db.upsert_employees(frame, 'initial.csv')
    changed = frame.copy()
    changed['NationalID'] = ['SECRET-A', 'SECRET-B']
    changed['BankAccount'] = ['IBAN-A', 'IBAN-B']
    result = engine.preview_merge(changed)
    assert result.updated == 0
    payload = json.dumps([item.to_dict() for item in result.employee_changes], allow_nan=False)
    assert 'NationalID' not in payload and 'BankAccount' not in payload
    assert 'SECRET-' not in payload and 'IBAN-' not in payload


def test_governed_sensitive_field_marks_change_without_raw_value_disclosure(engine, db):
    frame = workforce()
    db.upsert_employees(frame, 'initial.csv')
    changed = frame.copy()
    changed.loc[0, 'ManagerID'] = 'M999-SECRET'
    result = engine.preview_merge(changed)
    update = next(item for item in result.employee_changes if item.employee_id == 'E001')
    manager = next(change for change in update.changes if change.field_name == 'ManagerID')
    assert update.change_type == 'updated'
    assert manager.values_redacted is True
    assert manager.old_value is None and manager.new_value is None
    assert 'M999-SECRET' not in json.dumps(update.to_dict(), allow_nan=False)


def test_supported_non_sensitive_changes_remain_reviewable(engine, db):
    frame = workforce()
    db.upsert_employees(frame, 'initial.csv')
    changed = frame.copy()
    changed.loc[0, 'Salary'] = 80000.0
    result = engine.preview_merge(changed)
    update = next(item for item in result.employee_changes if item.employee_id == 'E001')
    salary = next(change for change in update.changes if change.field_name == 'Salary')
    assert salary.old_value == 70000.0 and salary.new_value == 80000.0
    assert salary.values_redacted is False


def test_supported_field_outside_legacy_compare_list_is_persisted(engine, db):
    frame = workforce()
    db.upsert_employees(frame, 'initial.csv')
    changed = frame.copy()
    changed.loc[0, 'JobTitle'] = 'Senior Partner'
    preview = engine.preview_merge(changed)
    assert preview.updated == 1
    result = engine.execute_merge(changed, 'job-title.csv')
    assert result.updated >= 1
    stored = db.get_all_employees().set_index('EmployeeID')
    assert stored.loc['E001', 'JobTitle'] == 'Senior Partner'


def test_preview_does_not_mutate_source(engine, db):
    frame = workforce()
    db.upsert_employees(frame, 'initial.csv')
    changed = frame.copy(deep=True)
    before = changed.copy(deep=True)
    engine.preview_merge(changed)
    pd.testing.assert_frame_equal(changed, before)


def test_snapshot_input_uses_latest_record_per_employee(engine, db):
    initial = workforce().iloc[[0]].copy()
    db.upsert_employees(initial, 'initial.csv')
    snapshots = pd.concat([initial, initial], ignore_index=True)
    snapshots['SnapshotDate'] = ['2026-01-01', '2026-02-01']
    snapshots.loc[0, 'Salary'] = 71000.0
    snapshots.loc[1, 'Salary'] = 72000.0
    result = engine.preview_merge(snapshots)
    assert result.total == 2
    assert result.updated == 1
    assert result.skipped == 1
    change = result.employee_changes[0].changes
    salary = next(item for item in change if item.field_name == 'Salary')
    assert salary.new_value == 72000.0


def test_conflicting_snapshot_ties_fail_closed(engine, db):
    initial = workforce().iloc[[0]].copy()
    db.upsert_employees(initial, 'initial.csv')
    snapshots = pd.concat([initial, initial], ignore_index=True)
    snapshots['SnapshotDate'] = ['2026-02-01', '2026-02-01']
    snapshots.loc[1, 'Salary'] = 99999.0
    with pytest.raises(ValueError, match='Conflicting employee records'):
        engine.preview_merge(snapshots)


@pytest.mark.parametrize('salary_threshold,rating_threshold', [
    (-0.1, 0.5), (0.1, -0.5), (np.nan, 0.5), (0.1, np.inf),
    (True, 0.5), (0.1, False), ('0.1', 0.5),
])
def test_significance_thresholds_must_be_finite_nonnegative_numbers(engine, salary_threshold, rating_threshold):
    with pytest.raises(ValueError):
        engine.get_significant_changes(engine.preview_merge(workforce()), salary_threshold, rating_threshold)


def test_nonfinite_numeric_change_never_escapes_as_json_number(engine, db):
    frame = workforce()
    db.upsert_employees(frame, 'initial.csv')
    changed = frame.copy()
    changed.loc[0, 'Salary'] = np.inf
    result = engine.preview_merge(changed)
    payload = json.dumps([item.to_dict() for item in result.employee_changes], allow_nan=False)
    assert 'Infinity' not in payload and 'NaN' not in payload


def test_execute_merge_writes_only_added_or_changed_rows(engine, db, monkeypatch):
    frame = workforce()
    db.upsert_employees(frame, 'initial.csv')
    changed = frame.copy()
    changed.loc[0, 'Salary'] = 80000.0
    observed = {}
    real_upsert = db.upsert_employees

    def recording_upsert(rows, file_name):
        observed['ids'] = rows['EmployeeID'].tolist()
        return real_upsert(rows, file_name)

    monkeypatch.setattr(db, 'upsert_employees', recording_upsert)
    result = engine.execute_merge(changed, 'delta.csv')
    assert observed['ids'] == ['E001']
    assert result.updated >= 1


def test_preview_summary_is_strict_json_serializable(engine, db):
    frame = workforce()
    db.upsert_employees(frame, 'initial.csv')
    changed = frame.copy()
    changed.loc[0, 'Salary'] = 81000.0
    result = engine.preview_merge(changed)
    json.dumps(result.to_summary_dict(), allow_nan=False)
    json.dumps([item.to_dict() for item in result.employee_changes], allow_nan=False)
