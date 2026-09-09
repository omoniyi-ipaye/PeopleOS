"""Cycle 007 deep mathematical and semantic engine-integrity validation.

These tests complement example-based known-answer tests with reconciliation,
metamorphic and determinism contracts. A failure here should be treated as an
engine-integrity defect rather than a renderer/test wording mismatch.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.agent.analysis_sandbox import AnalysisSpec, CohortFilter, GovernedAnalysisSandbox
from src.analytics_engine import AnalyticsEngine
from src.compensation_engine import CompensationEngine
from src.population import resolve_current_population


def workforce() -> pd.DataFrame:
    """Controlled current-state workforce with known outcomes and exclusions."""
    rows = []
    for i in range(36):
        dept = ['Engineering', 'Product', 'People'][i % 3]
        location = ['Madrid', 'London'][i % 2]
        attrition = 1 if i in {24, 25, 26, 27, 28, 29} else 0
        # Four current rows deliberately have unknown outcomes and therefore are
        # neither active nor part of the observed attrition denominator.
        if i >= 32:
            attrition = None
        rows.append({
            'EmployeeID': f'E{i:03d}',
            'Dept': dept,
            'Location': location,
            'JobLevel': ['L2', 'L3', 'L4'][i % 3],
            'JobTitle': 'Analyst' if i % 2 else 'Manager',
            'Gender': 'Female' if i % 2 else 'Male',
            'Salary': float(50_000 + (i % 6) * 10_000),
            'Tenure': float(i % 8),
            'Age': float(25 + (i % 20)),
            'LastRating': float(1 + (i % 5)),
            'Attrition': attrition,
            'SnapshotDate': '2026-09-01',
        })
    return pd.DataFrame(rows)


def sorted_records(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values('EmployeeID').reset_index(drop=True)


def test_reconciliation_accounting_identities_hold_across_summary_and_departments():
    frame = workforce()
    engine = AnalyticsEngine(frame)
    summary = engine.get_summary_statistics()
    departments = engine.get_department_aggregates()

    assert departments['Headcount'].sum() == summary['headcount']
    assert departments['Total_Records'].sum() == summary['record_count']
    assert departments['Outcome_Observations'].sum() == summary['attrition_known_count']
    assert summary['attrition_count'] + (summary['attrition_known_count'] - summary['attrition_count']) == summary['attrition_known_count']
    assert summary['attrition_known_count'] + summary['attrition_excluded_count'] == summary['record_count']

    weighted_departure_count = sum(
        row.Outcome_Observations * row.Observed_Attrition_Share
        for row in departments.itertuples()
        if row.Observed_Attrition_Share is not None and not pd.isna(row.Observed_Attrition_Share)
    )
    assert weighted_departure_count == pytest.approx(summary['attrition_count'])
    assert weighted_departure_count / summary['attrition_known_count'] == pytest.approx(summary['observed_attrition_share'])


def test_distribution_counts_reconcile_to_their_declared_populations():
    engine = AnalyticsEngine(workforce())
    summary = engine.get_summary_statistics()

    assert int(engine.get_tenure_distribution()['Count'].sum()) == summary['headcount']
    assert int(engine.get_age_distribution()['Count'].sum()) == summary['headcount']
    assert int(engine.get_salary_bands()['Count'].sum()) == summary['salary_observations']


def test_analytics_and_compensation_share_one_salary_truth():
    frame = workforce()
    analytics = AnalyticsEngine(frame).get_summary_statistics()
    compensation = CompensationEngine(frame).get_compensation_summary()

    assert analytics['salary_mean'] == pytest.approx(compensation['avg_salary'])
    assert analytics['salary_median'] == pytest.approx(compensation['median_salary'])
    assert analytics['salary_std'] == pytest.approx(compensation['std_dev'])
    assert analytics['salary_observations'] == compensation['salary_observations']
    assert analytics['salary_excluded_count'] == compensation['excluded_salary_count']
    assert analytics['headcount'] == compensation['active_count']
    assert compensation['total_payroll'] == pytest.approx(compensation['avg_salary'] * compensation['salary_observations'])


def test_payroll_changes_by_exactly_the_salary_delta_for_one_active_employee():
    frame = workforce()
    baseline = CompensationEngine(frame).get_compensation_summary()
    changed = frame.copy()
    target = changed.index[changed['Attrition'].eq(0)][0]
    changed.loc[target, 'Salary'] += 12_345.67
    after = CompensationEngine(changed).get_compensation_summary()

    assert after['headcount'] == baseline['headcount']
    assert after['total_payroll'] - baseline['total_payroll'] == pytest.approx(12_345.67)


def test_positive_salary_rescaling_preserves_dispersion_shape_and_scales_location():
    frame = workforce()
    base_engine = CompensationEngine(frame)
    base = base_engine.get_compensation_summary()
    base_dispersion = base_engine.calculate_pay_equity_score().sort_values('Dept').reset_index(drop=True)

    scaled = frame.copy()
    scaled['Salary'] = scaled['Salary'] * 3.5
    scaled_engine = CompensationEngine(scaled)
    scaled_summary = scaled_engine.get_compensation_summary()
    scaled_dispersion = scaled_engine.calculate_pay_equity_score().sort_values('Dept').reset_index(drop=True)

    assert scaled_summary['avg_salary'] == pytest.approx(base['avg_salary'] * 3.5)
    assert scaled_summary['median_salary'] == pytest.approx(base['median_salary'] * 3.5)
    assert scaled_summary['total_payroll'] == pytest.approx(base['total_payroll'] * 3.5)
    assert scaled_dispersion['CV'].to_numpy() == pytest.approx(base_dispersion['CV'].to_numpy())
    assert scaled_dispersion['Gini'].to_numpy() == pytest.approx(base_dispersion['Gini'].to_numpy())
    assert scaled_dispersion['SalaryDispersionScore'].to_numpy() == pytest.approx(base_dispersion['SalaryDispersionScore'].to_numpy())


def test_unknown_attrition_rows_do_not_move_observed_attrition_share():
    frame = workforce()
    baseline = AnalyticsEngine(frame).get_summary_statistics()

    extra = pd.DataFrame([{
        'EmployeeID': 'E999', 'Dept': 'Engineering', 'Location': 'Madrid', 'JobLevel': 'L3',
        'JobTitle': 'Analyst', 'Gender': 'Female', 'Salary': 99_000., 'Tenure': 3., 'Age': 31.,
        'LastRating': 4., 'Attrition': 'not recorded', 'SnapshotDate': '2026-09-01',
    }])
    after = AnalyticsEngine(pd.concat([frame, extra], ignore_index=True)).get_summary_statistics()

    assert after['record_count'] == baseline['record_count'] + 1
    assert after['attrition_known_count'] == baseline['attrition_known_count']
    assert after['observed_attrition_share'] == pytest.approx(baseline['observed_attrition_share'])
    assert after['headcount'] == baseline['headcount']  # unknown outcome is not silently active


def test_row_order_does_not_change_unique_employee_results():
    frame = workforce()
    baseline = AnalyticsEngine(frame).get_summary_statistics()
    shuffled = AnalyticsEngine(frame.sample(frac=1, random_state=73)).get_summary_statistics()

    keys = [
        'headcount', 'record_count', 'observed_attrition_share', 'department_count',
        'salary_mean', 'salary_median', 'salary_std', 'salary_observations',
        'tenure_mean', 'age_mean', 'lastrating_mean', 'attrition_count', 'attrition_known_count',
    ]
    for key in keys:
        assert shuffled[key] == pytest.approx(baseline[key]) if isinstance(baseline[key], float) else shuffled[key] == baseline[key]


def test_historical_rows_do_not_double_count_current_employee_state():
    current = workforce()
    history = current.copy()
    history['SnapshotDate'] = '2026-08-01'
    history['Salary'] = history['Salary'] - 5_000
    history['Attrition'] = 0
    combined = pd.concat([history, current], ignore_index=True)

    current_summary = AnalyticsEngine(current).get_summary_statistics()
    combined_summary = AnalyticsEngine(combined).get_summary_statistics()

    for key in ['headcount', 'record_count', 'observed_attrition_share', 'salary_mean', 'salary_median', 'attrition_count', 'attrition_known_count']:
        if isinstance(current_summary[key], float):
            assert combined_summary[key] == pytest.approx(current_summary[key])
        else:
            assert combined_summary[key] == current_summary[key]
    assert combined_summary['snapshot_history'] is True
    assert combined_summary['population_as_of_date'] == '2026-09-01'


def test_equal_timestamp_conflicts_resolve_deterministically_independent_of_source_order():
    base = workforce().iloc[:10].copy()
    conflict_a = base.iloc[[0]].copy()
    conflict_b = base.iloc[[0]].copy()
    conflict_a['Salary'] = 60_000.
    conflict_b['Salary'] = 160_000.
    conflict_a['SnapshotDate'] = '2026-09-05'
    conflict_b['SnapshotDate'] = '2026-09-05'

    first = pd.concat([base, conflict_a, conflict_b], ignore_index=True)
    second = pd.concat([base, conflict_b, conflict_a], ignore_index=True)
    resolved_first, _ = resolve_current_population(first)
    resolved_second, _ = resolve_current_population(second)

    pd.testing.assert_frame_equal(sorted_records(resolved_first), sorted_records(resolved_second), check_dtype=False)


def test_governed_group_summary_matches_manual_filtered_oracle():
    frame = workforce()
    spec = AnalysisSpec(
        operation='group_summary', population='active', group_by='JobLevel', measure='Salary', statistic='mean',
        filters=[CohortFilter(column='Dept', operator='eq', value='Engineering')],
    )
    result = GovernedAnalysisSandbox(frame).run(spec)
    assert result['available'] is True

    active = AnalyticsEngine(frame).active_df
    manual = active[active['Dept'].eq('Engineering')].groupby('JobLevel')['Salary'].agg(['mean', 'count'])
    returned = {row['group']: row for row in result['output']['groups']}
    for level, row in manual.iterrows():
        if row['count'] >= 5:
            assert returned[level]['value'] == pytest.approx(row['mean'])
            assert returned[level]['measured_count'] == row['count']


def test_governed_compare_groups_honors_requested_sum_statistic():
    frame = workforce()
    spec = AnalysisSpec(
        operation='compare_groups', population='active', group_by='Location', measure='Salary',
        statistic='sum', group_a='Madrid', group_b='London',
    )
    result = GovernedAnalysisSandbox(frame).run(spec)
    assert result['available'] is True

    active = AnalyticsEngine(frame).active_df
    expected = active.groupby('Location')['Salary'].sum()
    returned = {row['group']: row['value'] for row in result['output']['groups']}
    assert returned['Madrid'] == pytest.approx(expected['Madrid'])
    assert returned['London'] == pytest.approx(expected['London'])
    assert result['output']['difference_b_minus_a'] == pytest.approx(expected['London'] - expected['Madrid'])


def test_governed_correlation_is_order_invariant_and_affine_invariant():
    frame = workforce().copy()
    # Make the relation non-constant and supported while keeping active records.
    frame['Tenure'] = np.arange(len(frame), dtype=float)
    frame['Salary'] = 40_000 + frame['Tenure'] * 1_250
    spec = AnalysisSpec(operation='correlation', population='active', measure='Tenure', second_measure='Salary')

    base = GovernedAnalysisSandbox(frame).run(spec)
    shuffled = GovernedAnalysisSandbox(frame.sample(frac=1, random_state=19)).run(spec)
    transformed = frame.copy()
    transformed['Salary'] = transformed['Salary'] * 2 + 100_000
    affine = GovernedAnalysisSandbox(transformed).run(spec)

    assert base['available'] and shuffled['available'] and affine['available']
    assert shuffled['output']['correlation'] == pytest.approx(base['output']['correlation'])
    assert affine['output']['correlation'] == pytest.approx(base['output']['correlation'])
    assert shuffled['output']['paired_observations'] == base['output']['paired_observations']
    assert affine['output']['paired_observations'] == base['output']['paired_observations']


def test_governed_analysis_never_serializes_nan_or_infinity():
    frame = workforce().copy()
    frame.loc[frame.index[:3], 'Salary'] = [np.nan, np.inf, -np.inf]
    spec = AnalysisSpec(operation='group_summary', population='active', group_by='Dept', measure='Salary', statistic='mean')
    result = GovernedAnalysisSandbox(frame).run(spec)
    json.dumps(result, allow_nan=False)


def test_identifier_like_columns_are_blocked_even_as_filters():
    spec = AnalysisSpec(
        operation='group_summary', population='active', group_by='Dept', statistic='count',
        filters=[CohortFilter(column='EmployeeID', operator='eq', value='E001')],
    )
    with pytest.raises(ValueError, match='Identifier-like'):
        GovernedAnalysisSandbox(workforce()).run(spec)
