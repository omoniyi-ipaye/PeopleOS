"""Cycle 008 forensic contracts for AnalyticsEngine only."""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats

from src.analytics_engine import AnalyticsEngine


def frame(n: int = 60) -> pd.DataFrame:
    return pd.DataFrame({
        'EmployeeID': [f'E{i:04d}' for i in range(n)],
        'Dept': np.resize(['A', 'B', 'C'], n),
        'Tenure': np.arange(n, dtype=float) % 12,
        'Salary': 50_000.0 + (np.arange(n, dtype=float) % 10) * 5_000,
        'LastRating': 1.0 + (np.arange(n, dtype=float) % 5),
        'Age': 20.0 + (np.arange(n, dtype=float) % 40),
        'Attrition': np.resize([0, 0, 0, 1], n),
    })


def test_summary_matches_independent_plain_arithmetic():
    df = frame(64)
    df.loc[0, 'Salary'] = -1
    df.loc[1, 'Salary'] = np.inf
    df.loc[2, 'Age'] = 999
    df.loc[3, 'LastRating'] = 9
    df.loc[4, 'Tenure'] = -2
    engine = AnalyticsEngine(df)
    summary = engine.get_summary_statistics()
    active = df[df['Attrition'].eq(0)]
    valid_salary = [float(x) for x in active['Salary'] if np.isfinite(x) and x > 0]
    valid_age = [float(x) for x in active['Age'] if np.isfinite(x) and 1 <= x <= 120]
    valid_rating = [float(x) for x in active['LastRating'] if np.isfinite(x) and 1 <= x <= 5]
    valid_tenure = [float(x) for x in active['Tenure'] if np.isfinite(x) and x >= 0]
    assert summary['headcount'] == len(active)
    assert summary['record_count'] == len(df)
    assert summary['salary_mean'] == pytest.approx(sum(valid_salary) / len(valid_salary))
    assert summary['salary_observations'] == len(valid_salary)
    assert summary['age_observations'] == len(valid_age)
    assert summary['lastrating_observations'] == len(valid_rating)
    assert summary['tenure_observations'] == len(valid_tenure)
    assert summary['attrition_count'] == int(df['Attrition'].sum())
    assert summary['attrition_known_count'] == len(df)


def test_department_aggregates_reconcile_every_population_and_salary_mean():
    df = frame(90)
    df.loc[[2, 7], 'Salary'] = [0, np.nan]
    result = AnalyticsEngine(df).get_department_aggregates().set_index('Dept')
    assert int(result['Headcount'].sum()) == int((df['Attrition'] == 0).sum())
    assert int(result['Total_Records'].sum()) == len(df)
    assert int(result['Outcome_Observations'].sum()) == len(df)
    for dept, part in df.groupby('Dept'):
        active = part[part['Attrition'].eq(0)]
        salary = pd.to_numeric(active['Salary'], errors='coerce')
        salary = salary[np.isfinite(salary) & (salary > 0)]
        expected = None if salary.empty else float(salary.mean())
        actual = result.loc[dept, 'Avg_Salary']
        if expected is None:
            assert pd.isna(actual)
        else:
            assert actual == pytest.approx(expected)


def test_observed_attrition_ignores_unknown_labels_without_changing_headcount():
    df = frame(40)
    baseline = AnalyticsEngine(df).get_summary_statistics()
    extra = pd.DataFrame([{
        'EmployeeID': 'UNKNOWN', 'Dept': 'A', 'Tenure': 2, 'Salary': 70_000,
        'LastRating': 3, 'Age': 32, 'Attrition': 'not-known'
    }])
    changed = AnalyticsEngine(pd.concat([df, extra], ignore_index=True)).get_summary_statistics()
    assert changed['observed_attrition_share'] == pytest.approx(baseline['observed_attrition_share'])
    assert changed['attrition_known_count'] == baseline['attrition_known_count']
    assert changed['attrition_excluded_count'] == baseline['attrition_excluded_count'] + 1
    assert changed['headcount'] == baseline['headcount']


@pytest.mark.parametrize('value,expected', [
    (-1, 'Unknown'), (0, '<1 year'), (0.9999, '<1 year'), (1, '1-2 years'),
    (1.9999, '1-2 years'), (2, '2-5 years'), (4.9999, '2-5 years'),
    (5, '5-10 years'), (9.9999, '5-10 years'), (10, '10+ years'),
])
def test_tenure_bucket_boundaries(value, expected):
    df = frame(1)
    df.loc[0, 'Tenure'] = value
    dist = AnalyticsEngine(df).get_tenure_distribution().set_index('Tenure_Range')['Count']
    assert int(dist.loc[expected]) == 1
    assert int(dist.sum()) == 1


@pytest.mark.parametrize('value,expected', [
    (0, 'Unknown'), (1, 'Under 25'), (24.9999, 'Under 25'), (25, '25-34'),
    (34.9999, '25-34'), (35, '35-44'), (44.9999, '35-44'),
    (45, '45-54'), (54.9999, '45-54'), (55, '55+'), (120, '55+'), (121, 'Unknown'),
])
def test_age_bucket_boundaries(value, expected):
    df = frame(1)
    df.loc[0, 'Age'] = value
    dist = AnalyticsEngine(df).get_age_distribution().set_index('Age_Range')['Count']
    assert int(dist.loc[expected]) == 1
    assert int(dist.sum()) == 1


def test_salary_bands_cover_each_valid_salary_exactly_once_even_with_ties():
    df = frame(32)
    df['Attrition'] = 0
    df['Salary'] = [50_000] * 8 + [60_000] * 8 + [70_000] * 8 + [80_000] * 8
    bands = AnalyticsEngine(df).get_salary_bands()
    assert int(bands['Count'].sum()) == 32
    assert (bands['Count'] >= 0).all()
    assert (bands['Upper'] >= bands['Lower']).all()


def test_salary_bands_all_equal_still_reconcile_population():
    df = frame(20)
    df['Attrition'] = 0
    df['Salary'] = 75_000.0
    bands = AnalyticsEngine(df).get_salary_bands()
    assert int(bands['Count'].sum()) == 20
    assert int((bands['Count'] > 0).sum()) == 1


def test_correlation_matches_scipy_reference_and_pairwise_support():
    n = 40
    df = frame(n)
    df['Attrition'] = np.resize([0, 1], n)
    df['Tenure'] = np.arange(n, dtype=float)
    df.loc[:4, 'Tenure'] = np.nan
    result = AnalyticsEngine(df).get_correlations('Attrition').set_index('Feature')
    pair = df[['Tenure', 'Attrition']].dropna()
    r, p = scipy_stats.pearsonr(pair['Tenure'], pair['Attrition'])
    assert result.loc['Tenure', 'Correlation'] == pytest.approx(r)
    assert result.loc['Tenure', 'P_Value'] == pytest.approx(p)
    assert int(result.loc['Tenure', 'Observations']) == len(pair)


def test_correlation_is_invariant_to_row_order_and_positive_affine_transform():
    df = frame(80)
    df['Attrition'] = np.resize([0, 1], len(df))
    df['Tenure'] = np.arange(len(df), dtype=float)
    base = AnalyticsEngine(df).get_correlations('Attrition').set_index('Feature').loc['Tenure', 'Correlation']
    shuffled = AnalyticsEngine(df.sample(frac=1, random_state=42)).get_correlations('Attrition').set_index('Feature').loc['Tenure', 'Correlation']
    scaled = df.copy(); scaled['Tenure'] = scaled['Tenure'] * 7 + 100
    transformed = AnalyticsEngine(scaled).get_correlations('Attrition').set_index('Feature').loc['Tenure', 'Correlation']
    assert shuffled == pytest.approx(base)
    assert transformed == pytest.approx(base)


def test_correlation_below_minimum_support_is_omitted():
    df = frame(12)
    df['Attrition'] = np.resize([0, 1], 12)
    df['Tenure'] = np.nan
    df.loc[:8, 'Tenure'] = np.arange(9)
    result = AnalyticsEngine(df).get_correlations('Attrition')
    assert 'Tenure' not in result.get('Feature', pd.Series(dtype=str)).tolist()


def test_two_group_comparison_matches_welch_reference():
    df = frame(40)
    df['Attrition'] = 0
    df['Dept'] = ['A'] * 20 + ['B'] * 20
    df['Salary'] = list(np.arange(20) + 50_000) + list(np.arange(20) + 51_000)
    actual = AnalyticsEngine(df).compare_groups('Dept', 'Salary')
    stat, p = scipy_stats.ttest_ind(df.loc[df.Dept == 'A', 'Salary'], df.loc[df.Dept == 'B', 'Salary'], equal_var=False)
    assert actual['success']
    assert actual['test_name'] == "Welch's T-Test"
    assert actual['statistic'] == pytest.approx(stat)
    assert actual['p_value'] == pytest.approx(p)
    assert actual['sample_size'] == 40


def test_three_group_comparison_matches_anova_reference():
    df = frame(45)
    df['Attrition'] = 0
    df['Dept'] = ['A'] * 15 + ['B'] * 15 + ['C'] * 15
    df['Salary'] = np.concatenate([np.arange(15), np.arange(15) + 3, np.arange(15) + 6]) + 50_000
    actual = AnalyticsEngine(df).compare_groups('Dept', 'Salary')
    groups = [df.loc[df.Dept == name, 'Salary'] for name in ['A', 'B', 'C']]
    stat, p = scipy_stats.f_oneway(*groups)
    assert actual['success']
    assert actual['test_name'] == 'One-way ANOVA'
    assert actual['statistic'] == pytest.approx(stat)
    assert actual['p_value'] == pytest.approx(p)


def test_group_comparison_uses_valid_measured_rows_for_minimum_support():
    df = frame(22)
    df['Attrition'] = 0
    df['Dept'] = ['A'] * 11 + ['B'] * 11
    df.loc[0:1, 'Salary'] = np.nan  # A now has only 9 measured rows.
    assert AnalyticsEngine(df).compare_groups('Dept', 'Salary')['success'] is False


def test_confidence_interval_matches_scipy_t_interval():
    df = frame(30)
    df['Attrition'] = 0
    values = df['Salary'].astype(float)
    mean = values.mean(); sem = scipy_stats.sem(values)
    expected = scipy_stats.t.interval(0.95, len(values) - 1, loc=mean, scale=sem)
    actual = AnalyticsEngine(df).get_confidence_interval('Salary', 0.95)
    assert actual[0] == pytest.approx(expected[0])
    assert actual[1] == pytest.approx(expected[1])


def test_confidence_interval_constant_values_is_point_interval():
    df = frame(20); df['Attrition'] = 0; df['Salary'] = 100_000.0
    assert AnalyticsEngine(df).get_confidence_interval('Salary') == pytest.approx((100_000.0, 100_000.0))


def test_confidence_interval_never_emits_nonfinite_bounds_for_extreme_finite_inputs():
    df = frame(20); df['Attrition'] = 0
    df['Salary'] = [1e308, 9e307] * 10
    result = AnalyticsEngine(df).get_confidence_interval('Salary')
    assert result is None or all(math.isfinite(float(x)) for x in result)


@pytest.mark.parametrize('threshold', [-0.01, 1.01, float('nan'), float('inf'), -float('inf')])
def test_high_risk_department_threshold_rejects_invalid_share_thresholds(threshold):
    with pytest.raises(ValueError):
        AnalyticsEngine(frame()).get_high_risk_departments(threshold)


def test_high_risk_department_boundary_is_strictly_greater_than_threshold():
    df = frame(40)
    df['Dept'] = ['A'] * 20 + ['B'] * 20
    df['Attrition'] = [1] * 4 + [0] * 16 + [1] * 5 + [0] * 15
    result = AnalyticsEngine(df).get_high_risk_departments(0.20)
    assert result['Dept'].tolist() == ['B']


def test_temporal_stats_use_current_population_and_finite_values_only():
    df = frame(8)
    df['RatingVelocity'] = [1, 2, np.inf, -np.inf, np.nan, 3, 4, 5]
    expected = np.mean([1, 2, 3, 4, 5])
    assert AnalyticsEngine(df).get_temporal_stats()['avg_velocity'] == pytest.approx(expected)


def test_source_dataframe_is_not_mutated():
    df = frame(30)
    original = df.copy(deep=True)
    engine = AnalyticsEngine(df)
    engine.get_summary_statistics(); engine.get_department_aggregates(); engine.get_correlations(); engine.get_tenure_distribution(); engine.get_age_distribution(); engine.get_salary_bands()
    pd.testing.assert_frame_equal(df, original)
