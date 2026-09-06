"""Regression tests for PeopleOS output-integrity contracts."""

import numpy as np
import pandas as pd

from src.analytics_engine import AnalyticsEngine
from src.compensation_engine import CompensationEngine
from src.fairness_engine import FairnessEngine
from src.population import active_population, normalize_attrition, observed_attrition_share, resolve_current_population
from src.preprocessor import Preprocessor


def _base_rows(n=40):
    return pd.DataFrame({
        'EmployeeID': [f'E{i:03d}' for i in range(n)],
        'Dept': ['A'] * (n // 2) + ['B'] * (n - n // 2),
        'Tenure': np.linspace(.5, 10, n),
        'Salary': np.linspace(50_000, 100_000, n),
        'LastRating': np.linspace(2.5, 4.5, n),
        'Age': np.linspace(24, 58, n),
        'Gender': ['Female', 'Male'] * (n // 2),
        'Attrition': [0] * (n - 8) + [1] * 8,
    })


def test_latest_snapshot_is_single_current_employee_observation():
    older = _base_rows(20)
    older['SnapshotDate'] = '2025-12-31'
    newer = older.copy()
    newer['SnapshotDate'] = '2026-06-30'
    newer.loc[0, 'Salary'] = 123_456
    history = pd.concat([older, newer], ignore_index=True)

    current, resolution = resolve_current_population(history)
    assert len(current) == 20
    assert resolution.source_rows == 40
    assert resolution.snapshot_history is True
    assert resolution.as_of_date == '2026-06-30'
    assert current.loc[current.EmployeeID == 'E000', 'Salary'].iloc[0] == 123_456


def test_attrition_normalization_fails_unknown_labels_to_missing():
    values = pd.Series(['Active', 'left', '0', '1', 'maybe'])
    normalized = normalize_attrition(values)
    assert normalized.iloc[:4].tolist() == [0, 1, 0, 1]
    assert pd.isna(normalized.iloc[4])


def test_headcount_is_active_population_not_record_count():
    data = _base_rows(40)
    engine = AnalyticsEngine(data)
    assert engine.get_record_count() == 40
    assert engine.get_headcount() == 32
    assert len(active_population(data)) == 32
    assert engine.get_observed_attrition_share() == 0.2
    summary = engine.get_summary_statistics()
    assert summary['headcount'] == 32
    assert summary['record_count'] == 40
    assert summary['turnover_rate_semantics'] == 'observed_attrition_share_not_period_turnover'


def test_current_compensation_excludes_departed_and_invalid_salary():
    data = _base_rows(40)
    data.loc[0, 'Salary'] = -1
    # Departed rows carry intentionally huge historic salaries and must not enter current payroll.
    data.loc[data['Attrition'] == 1, 'Salary'] = 1_000_000
    engine = CompensationEngine(data)
    summary = engine.get_compensation_summary()
    assert summary['headcount'] == 31
    assert summary['max_salary'] < 1_000_000
    assert summary['population'] == 'current_active_employees_with_valid_positive_salary'
    equity = engine.calculate_pay_equity_score()
    assert (equity['MetricSemantics'] == 'salary_dispersion_consistency_not_adjusted_pay_equity').all()


def test_four_fifths_uses_favorable_retention_rate_and_suppresses_small_groups():
    # Group A attrition 10% => retention 90%; B attrition 40% => retention 60%.
    a = pd.DataFrame({'EmployeeID': [f'A{i}' for i in range(20)], 'Gender': ['A'] * 20, 'Attrition': [1] * 2 + [0] * 18})
    b = pd.DataFrame({'EmployeeID': [f'B{i}' for i in range(20)], 'Gender': ['B'] * 20, 'Attrition': [1] * 8 + [0] * 12})
    tiny = pd.DataFrame({'EmployeeID': [f'C{i}' for i in range(4)], 'Gender': ['C'] * 4, 'Attrition': [0] * 4})
    engine = FairnessEngine(pd.concat([a, b, tiny], ignore_index=True))
    result = engine.calculate_four_fifths_rule('Attrition', favorable=False)
    gender = result[result['attribute'] == 'Gender'].set_index('group')
    assert 'C' not in gender.index
    assert abs(gender.loc['A', 'favorable_rate'] - .9) < 1e-9
    assert abs(gender.loc['B', 'favorable_rate'] - .6) < 1e-9
    assert abs(gender.loc['B', 'adverse_impact_ratio'] - (2 / 3)) < 1e-9
    assert bool(gender.loc['B', 'passes_4_5_rule']) is False


def test_preprocessor_reuses_training_statistics_and_unknown_category():
    train = _base_rows(30).drop(columns=['Attrition'])
    train['Dept'] = ['A'] * 15 + ['B'] * 15
    train.loc[0, 'Salary'] = np.nan
    p = Preprocessor()
    fitted, metadata = p.fit_transform(train, target_column='__no_target__')
    assert metadata['fit_scope'] == 'training_population'
    train_salary_median = p.impute_values['Salary']

    holdout = train.iloc[:3].copy()
    holdout['Dept'] = ['NEW', 'NEW', 'NEW']
    holdout['Salary'] = [np.nan, np.nan, np.nan]
    transformed = p.transform(holdout, target_column='__no_target__')
    assert p.impute_values['Salary'] == train_salary_median
    assert transformed['Dept'].notna().all()
