"""Independent input contracts: preserve people, refuse ambiguous history/pay."""
import numpy as np
import pandas as pd
import pytest

from src.analytics_engine import AnalyticsEngine
from src.data_loader import DataLoader, DataValidationError


def workforce():
    return pd.DataFrame({
        'EmployeeID': [f'E{i}' for i in range(50)], 'Dept': ['People'] * 50,
        'Salary': [60000.] * 50, 'Tenure': [3.] * 50, 'Age': [30.] * 50,
        'LastRating': [4.] * 50, 'Gender': ['Female', 'Male'] * 25,
        'JobTitle': ['Analyst'] * 50, 'Location': ['Madrid'] * 50,
        'HireDate': ['2023-01-01'] * 50, 'ManagerID': ['M1'] * 50,
        'Attrition': [0] * 49 + [1],
    })


def load(frame, tmp_path):
    path = tmp_path / 'workforce.csv'
    frame.to_csv(path, index=False)
    loader = DataLoader()
    return loader, loader.load(str(path))


@pytest.mark.parametrize('column', ['Salary', 'Tenure', 'Age'])
@pytest.mark.parametrize('invalid', [-1., np.inf, -np.inf, 'not measured'])
def test_bad_measurement_never_removes_a_person(column, invalid, tmp_path):
    frame = workforce()
    frame[column] = frame[column].astype(object)
    frame.loc[0, column] = invalid
    loader, result = load(frame, tmp_path)
    assert result.EmployeeID.tolist() == frame.EmployeeID.tolist()
    assert AnalyticsEngine(result).get_headcount() == 49
    assert pd.isna(result.loc[0, column])
    assert result.loc[1, column] == frame.loc[1, column]
    assert any(f'invalid {column}' in warning and 'preserved' in warning
               for warning in loader.validation_warnings)


@pytest.mark.parametrize('bad_date', ['not-a-date', '', None, '2026-02-30'])
def test_unknown_snapshot_order_rejects_entire_upload(bad_date, tmp_path):
    frame = workforce().assign(SnapshotDate='2026-01-01')
    newer = frame.iloc[[0]].copy()
    newer['SnapshotDate'] = bad_date
    newer['Attrition'] = 1
    with pytest.raises(DataValidationError, match='SnapshotDate'):
        load(pd.concat([frame, newer], ignore_index=True), tmp_path)


def test_valid_history_preserves_latest_outcome(tmp_path):
    frame = workforce().assign(SnapshotDate='2026-01-01')
    newer = frame.iloc[[0]].assign(SnapshotDate='2026-02-01', Attrition=1)
    _, result = load(pd.concat([frame, newer], ignore_index=True), tmp_path)
    assert len(result) == 51
    assert AnalyticsEngine(result).get_headcount() == 48


@pytest.mark.parametrize('column', ['PayPeriod', 'PayFrequency', 'pay_period', 'salary_frequency'])
@pytest.mark.parametrize('value', ['monthly', 'hourly', '', 'unknown'])
def test_incomparable_pay_rejected(column, value, tmp_path):
    frame = workforce()
    frame[column] = 'annual'
    frame.loc[0, column] = value
    frame.loc[0, 'Salary'] = 5000
    with pytest.raises(DataValidationError, match='annual salary'):
        load(frame, tmp_path)


@pytest.mark.parametrize('value', ['USD', '', None, '$'])
def test_mixed_or_incomplete_currency_rejected(value, tmp_path):
    frame = workforce().assign(Currency='EUR')
    frame.loc[0, 'Currency'] = value
    with pytest.raises(DataValidationError, match='Currency'):
        load(frame, tmp_path)


def test_explicit_comparable_pay_preserves_known_answer(tmp_path):
    frame = workforce().assign(pay_period=' Yearly ', salary_currency=' eur ')
    _, result = load(frame, tmp_path)
    assert result.Salary.sum() == 3000000
    assert result.Currency.unique().tolist() == ['EUR']
    assert len(result) == 50


def test_legacy_pay_assumptions_are_disclosed(tmp_path):
    loader, _ = load(workforce(), tmp_path)
    assert any('annual amounts' in warning for warning in loader.validation_warnings)
    assert any('one shared currency' in warning for warning in loader.validation_warnings)


def test_ambiguous_aliases_rejected(tmp_path):
    frame = workforce().assign(annual_salary=120000)
    with pytest.raises(DataValidationError, match="Multiple columns map to 'Salary'"):
        load(frame, tmp_path)


def test_sparse_pay_metadata_cannot_be_silently_dropped(tmp_path):
    frame = workforce().assign(Currency=None)
    frame.loc[0, 'Currency'] = 'EUR'
    with pytest.raises(DataValidationError, match='Currency'):
        load(frame, tmp_path)
