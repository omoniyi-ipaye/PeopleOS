"""Cycle 009 forensic contracts for CompensationEngine."""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats

from src.compensation_engine import CompensationEngine, CompensationEngineError, _gini


def frame(n: int = 80) -> pd.DataFrame:
    idx = np.arange(n)
    return pd.DataFrame({
        'EmployeeID': [f'E{i:05d}' for i in idx],
        'Dept': np.resize(['Engineering', 'Product', 'Sales', 'People'], n),
        'JobTitle': np.resize(['Engineer', 'PM', 'AE', 'Partner'], n),
        'Gender': np.resize(['Male', 'Female'], n),
        'Tenure': (idx % 15).astype(float),
        'Salary': 50_000.0 + (idx % 20) * 2_500.0,
        'LastRating': 1.0 + (idx % 5),
        'Age': 22.0 + (idx % 40),
        'Attrition': np.resize([0, 0, 0, 1], n),
    })


def active_valid_salary(df: pd.DataFrame) -> pd.Series:
    salary = pd.to_numeric(df.loc[pd.to_numeric(df['Attrition'], errors='coerce').eq(0), 'Salary'], errors='coerce')
    return salary[np.isfinite(salary) & (salary > 0)]


def test_summary_matches_plain_arithmetic_and_coverage():
    df = frame(40)
    df.loc[0, 'Salary'] = 0
    df.loc[1, 'Salary'] = -100
    df.loc[2, 'Salary'] = np.nan
    engine = CompensationEngine(df)
    result = engine.get_compensation_summary()
    salary = active_valid_salary(df)
    active_n = int((df['Attrition'] == 0).sum())
    assert result['headcount'] == len(salary)
    assert result['active_count'] == active_n
    assert result['salary_observations'] == len(salary)
    assert result['excluded_salary_count'] == active_n - len(salary)
    assert result['salary_coverage'] == pytest.approx(len(salary) / active_n)
    assert result['avg_salary'] == pytest.approx(float(salary.mean()))
    assert result['median_salary'] == pytest.approx(float(salary.median()))
    assert result['total_payroll'] == pytest.approx(float(salary.sum()))
    assert result['min_salary'] == float(salary.min())
    assert result['max_salary'] == float(salary.max())


def test_departed_salaries_do_not_change_current_compensation_summary():
    df = frame(60)
    engine = CompensationEngine(df)
    baseline = engine.get_compensation_summary()
    changed = df.copy()
    changed.loc[changed['Attrition'] == 1, 'Salary'] *= 100
    assert CompensationEngine(changed).get_compensation_summary() == pytest.approx(baseline)


def test_salary_percentiles_match_pandas_reference_per_department():
    df = frame(80)
    result = CompensationEngine(df).calculate_salary_percentiles().set_index('Dept')
    active = df[df['Attrition'].eq(0)]
    for dept, group in active.groupby('Dept'):
        salary = group['Salary'].astype(float)
        assert result.loc[dept, 'Headcount'] == len(salary)
        for q, col in [(0.10,'P10'), (0.25,'P25'), (0.50,'P50'), (0.75,'P75'), (0.90,'P90')]:
            assert result.loc[dept, col] == pytest.approx(salary.quantile(q))
        assert result.loc[dept, 'Mean'] == pytest.approx(salary.mean())


def test_gini_known_answers_and_scale_invariance():
    assert _gini(np.array([10, 10, 10, 10], dtype=float)) == pytest.approx(0.0)
    assert _gini(np.array([0, 0, 0, 100], dtype=float)) == pytest.approx(0.75)
    base = _gini(np.array([10, 20, 30, 40], dtype=float))
    assert _gini(np.array([100, 200, 300, 400], dtype=float)) == pytest.approx(base)


def test_dispersion_score_is_currency_scale_invariant():
    df = frame(80); df['Attrition'] = 0
    base = CompensationEngine(df).calculate_pay_equity_score().set_index('Dept')
    scaled = df.copy(); scaled['Salary'] *= 7.3
    transformed = CompensationEngine(scaled).calculate_pay_equity_score().set_index('Dept')
    for dept in base.index:
        assert transformed.loc[dept, 'CV'] == pytest.approx(base.loc[dept, 'CV'])
        assert transformed.loc[dept, 'Gini'] == pytest.approx(base.loc[dept, 'Gini'])
        assert transformed.loc[dept, 'SalaryDispersionScore'] == pytest.approx(base.loc[dept, 'SalaryDispersionScore'])
        assert transformed.loc[dept, 'AvgSalary'] == pytest.approx(base.loc[dept, 'AvgSalary'] * 7.3)


def test_salary_bands_reconcile_valid_population_with_ties():
    df = frame(40); df['Attrition'] = 0
    df['Salary'] = [50_000]*10 + [60_000]*10 + [70_000]*10 + [80_000]*10
    bands = CompensationEngine(df).get_salary_bands()
    assert int(bands['Count'].sum()) == 40
    assert (bands['Upper'] >= bands['Lower']).all()


def test_salary_bands_all_equal_count_each_salary_once():
    df = frame(32); df['Attrition'] = 0; df['Salary'] = 75_000.0
    bands = CompensationEngine(df).get_salary_bands()
    assert int(bands['Count'].sum()) == 32
    assert int((bands['Count'] > 0).sum()) == 1


def test_outlier_z_scores_match_manual_reference():
    df = frame(30); df['Attrition'] = 0; df['Dept'] = 'A'
    df['Salary'] = 50_000 + np.arange(30, dtype=float) * 100
    df.loc[29, 'Salary'] = 80_000
    out = CompensationEngine(df).identify_salary_outliers(2.5)
    salary = df['Salary']
    expected_z = (salary.iloc[29] - salary.mean()) / salary.std(ddof=1)
    row = out.loc[out['EmployeeID'].eq('E00029')].iloc[0]
    assert row['ZScore'] == pytest.approx(expected_z)
    assert row['Flag'] == 'Above department distribution'


@pytest.mark.parametrize('threshold', [0, -1, float('nan'), float('inf'), -float('inf')])
def test_outlier_threshold_must_be_finite_and_positive(threshold):
    with pytest.raises(ValueError):
        CompensationEngine(frame()).identify_salary_outliers(threshold)


def test_relative_compa_ratio_is_exactly_salary_over_department_median():
    df = frame(40); df['Attrition'] = 0
    result = CompensationEngine(df).calculate_compa_ratio()
    for dept, group in result.groupby('Dept'):
        midpoint = float(df.loc[df['Dept'].eq(dept), 'Salary'].median())
        assert np.allclose(group['BandMidpoint'], midpoint)
        assert np.allclose(group['CompaRatio'], group['Salary'] / midpoint)
        assert set(group['MetricSemantics']) == {'relative_to_department_median_not_formal_compa_ratio'}


def test_supplied_compa_ratio_reconstructs_finite_band_midpoint():
    df = frame(30); df['Attrition'] = 0
    df['CompaRatio'] = np.linspace(0.8, 1.2, len(df))
    result = CompensationEngine(df).calculate_compa_ratio()
    assert np.allclose(result['BandMidpoint'], result['Salary'] / result['CompaRatio'])
    assert np.isfinite(result['BandMidpoint']).all()
    assert set(result['MetricSemantics']) == {'supplied_compa_ratio'}


def test_supplied_compa_ratio_cannot_emit_nonfinite_band_midpoint():
    df = frame(30); df['Attrition'] = 0
    df['Salary'] = 1e200
    df['CompaRatio'] = 1e-200
    result = CompensationEngine(df).calculate_compa_ratio()
    numeric = pd.to_numeric(result['BandMidpoint'], errors='coerce')
    assert numeric.isna().all() or np.isfinite(numeric.dropna()).all()


def test_salary_attrition_association_matches_point_biserial_reference():
    df = frame(80)
    df['Salary'] = 50_000 + np.arange(len(df), dtype=float) * 750
    result = CompensationEngine(df).correlate_salary_with_attrition()
    valid = df['Attrition'].isin([0,1]) & (df['Salary'] > 0)
    r, p = scipy_stats.pointbiserialr(df.loc[valid, 'Attrition'].astype(int), df.loc[valid, 'Salary'].astype(float))
    assert result['available']
    assert result['correlation'] == pytest.approx(r)
    assert result['p_value'] == pytest.approx(p)
    assert result['sample_size'] == int(valid.sum())


def test_salary_attrition_requires_twenty_pairs_and_two_outcomes():
    df = frame(19)
    assert CompensationEngine(df).correlate_salary_with_attrition()['available'] is False
    df = frame(30); df['Attrition'] = 0
    assert CompensationEngine(df).correlate_salary_with_attrition()['available'] is False


def test_gender_pay_gap_matches_reference_denominator_and_welch_test():
    df = frame(40); df['Attrition'] = 0
    df['Gender'] = ['Male'] * 20 + ['Female'] * 20
    df['Salary'] = np.concatenate([np.arange(20)+100_000, np.arange(20)+90_000]).astype(float)
    result = CompensationEngine(df).calculate_gender_pay_gap(10)
    men = df.loc[df.Gender.eq('Male'), 'Salary']; women = df.loc[df.Gender.eq('Female'), 'Salary']
    expected_gap = (men.mean() - women.mean()) / men.mean() * 100
    t, p = scipy_stats.ttest_ind(men, women, equal_var=False)
    assert result['available']
    assert result['raw_gap_pct'] == pytest.approx(expected_gap)
    assert result['welch_t_stat'] == pytest.approx(t)
    assert result['p_value'] == pytest.approx(p)
    assert result['male_n'] == result['female_n'] == 20


@pytest.mark.parametrize('minimum', [0, -1, 1.5, float('nan'), float('inf')])
def test_gender_pay_gap_min_group_size_must_be_positive_integer(minimum):
    with pytest.raises(ValueError):
        CompensationEngine(frame()).calculate_gender_pay_gap(minimum)


def test_gender_pay_gap_fails_closed_without_two_supported_groups():
    df = frame(30); df['Attrition'] = 0; df['Gender'] = 'Unknown'
    result = CompensationEngine(df).calculate_gender_pay_gap(10)
    assert result['available'] is False


def test_constant_equal_gender_groups_have_descriptive_zero_gap_but_no_inference():
    df = frame(40); df['Attrition'] = 0
    df['Gender'] = ['Male']*20 + ['Female']*20; df['Salary'] = 100_000.0
    result = CompensationEngine(df).calculate_gender_pay_gap(10)
    assert result['available']
    assert result['raw_gap_pct'] == pytest.approx(0.0)
    assert result['inference_available'] is False
    assert result['p_value'] is None


def test_tenure_salary_bucket_boundaries_and_reconciliation():
    values = [-1, 0, .9999, 1, 1.9999, 2, 4.9999, 5, 9.9999, 10]
    expected = ['Unknown','<1 year','<1 year','1-2 years','1-2 years','2-5 years','2-5 years','5-10 years','5-10 years','10+ years']
    df = frame(len(values)); df['Attrition'] = 0; df['Tenure'] = values
    result = CompensationEngine(df).get_salary_by_tenure().set_index('TenureBucket')['Count']
    assert int(result.sum()) == len(values)
    for bucket in set(expected):
        assert int(result.loc[bucket]) == expected.count(bucket)


def test_row_order_does_not_change_aggregate_results():
    df = frame(120)
    a = CompensationEngine(df)
    b = CompensationEngine(df.sample(frac=1, random_state=99))
    for key in ['total_payroll','avg_salary','median_salary','min_salary','max_salary','salary_range','std_dev','headcount','salary_observations']:
        assert b.get_compensation_summary()[key] == pytest.approx(a.get_compensation_summary()[key])


def test_extreme_finite_inputs_never_escape_as_nonfinite_outputs():
    df = frame(20); df['Attrition'] = 0
    # Keep total payroll finite while challenging intermediate calculations.
    df['Salary'] = np.resize([1e306, 9e305], len(df))
    engine = CompensationEngine(df)
    summary = engine.get_compensation_summary()
    for value in summary.values():
        if isinstance(value, float):
            assert math.isfinite(value)
    dispersion = engine.calculate_pay_equity_score()
    for col in ['AvgSalary','StdDev','CV','Gini','EquityScore']:
        assert np.isfinite(pd.to_numeric(dispersion[col], errors='coerce').dropna()).all()


def test_source_dataframe_is_not_mutated():
    df = frame(60)
    original = df.copy(deep=True)
    engine = CompensationEngine(df)
    engine.get_compensation_summary(); engine.calculate_salary_percentiles(); engine.calculate_pay_equity_score()
    engine.identify_salary_outliers(); engine.get_salary_bands(); engine.calculate_compa_ratio()
    engine.correlate_salary_with_attrition(); engine.calculate_gender_pay_gap(); engine.get_salary_by_tenure()
    pd.testing.assert_frame_equal(df, original)
