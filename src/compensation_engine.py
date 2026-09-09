"""Compensation analytics with explicit population and metric semantics.

Current compensation measures use active employees with valid positive salary.
Historical/departed rows are retained only for explicitly historical association
analysis. The legacy `EquityScore` field is retained for API compatibility but
its canonical meaning is salary-dispersion consistency, not adjusted pay equity.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats

from src.logger import get_logger
from src.population import active_population, resolve_current_population
from src.utils import load_config

logger = get_logger('compensation_engine')


class CompensationEngineError(Exception):
    pass


def _gini(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0 or np.all(values == 0):
        return 0.0
    values = np.sort(values)
    if values.min() < 0:
        values = values - values.min()
    scale = values.max()
    if scale > 0:
        values = values / scale
    n = len(values)
    total = values.sum()
    if total == 0:
        return 0.0
    return float((2 * np.sum((np.arange(1, n + 1)) * values) / (n * total)) - (n + 1) / n)


def _stable_salary_stats(values: pd.Series) -> tuple[float, float, float]:
    """Calculate location and sample spread without overflowing intermediate sums."""
    array = values.to_numpy(dtype=float)
    if not len(array):
        raise CompensationEngineError('Salary statistics require at least one value')
    scale = float(np.max(np.abs(array)))
    if not np.isfinite(scale) or scale <= 0:
        raise CompensationEngineError('Salary magnitude exceeds the finite reporting range')
    scaled = array / scale
    mean = float(np.mean(scaled) * scale)
    ordered = np.sort(scaled)
    midpoint = len(ordered) // 2
    median_scaled = float(ordered[midpoint]) if len(ordered) % 2 else float(
        ordered[midpoint - 1] + (ordered[midpoint] - ordered[midpoint - 1]) / 2
    )
    median = float(median_scaled * scale)
    deviation = float(np.std(scaled, ddof=1) * scale) if len(array) > 1 else 0.0
    if not all(np.isfinite(value) for value in (mean, median, deviation)):
        raise CompensationEngineError('Salary magnitude exceeds the finite reporting range')
    return mean, median, deviation


def _stable_welch_ttest(first: pd.Series, second: pd.Series) -> tuple[Optional[float], Optional[float]]:
    """Welch t-test on a common positive scale to avoid overflow."""
    a = np.asarray(first, dtype=float)
    b = np.asarray(second, dtype=float)
    if len(a) < 2 or len(b) < 2 or not np.isfinite(a).all() or not np.isfinite(b).all():
        return None, None
    scale = float(max(np.max(np.abs(a)), np.max(np.abs(b))))
    if not np.isfinite(scale) or scale <= 0:
        return None, None
    a = a / scale
    b = b / scale
    mean_a, mean_b = float(np.mean(a)), float(np.mean(b))
    var_a, var_b = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
    if not all(np.isfinite(v) and v >= 0 for v in (var_a, var_b)):
        return None, None
    term_a = var_a / len(a)
    term_b = var_b / len(b)
    se2 = term_a + term_b
    if not np.isfinite(se2) or se2 <= 0:
        return None, None
    statistic = (mean_a - mean_b) / np.sqrt(se2)
    df_denom = (term_a * term_a) / (len(a) - 1) + (term_b * term_b) / (len(b) - 1)
    if not np.isfinite(statistic) or not np.isfinite(df_denom) or df_denom <= 0:
        return None, None
    degrees = (se2 * se2) / df_denom
    if not np.isfinite(degrees) or degrees <= 0:
        return None, None
    p_value = float(2 * stats.t.sf(abs(float(statistic)), degrees))
    if not np.isfinite(p_value):
        return None, None
    return float(statistic), p_value


def _positive_finite(value: Any, *, name: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be a finite positive number') from exc
    if not np.isfinite(numeric) or numeric <= 0:
        raise ValueError(f'{name} must be a finite positive number')
    return numeric


def _positive_integer(value: Any, *, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool):
        raise ValueError(f'{name} must be an integer of at least {minimum}')
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be an integer of at least {minimum}') from exc
    if not np.isfinite(numeric) or not numeric.is_integer() or numeric < minimum:
        raise ValueError(f'{name} must be an integer of at least {minimum}')
    return int(numeric)


class CompensationEngine:
    def __init__(self, df: pd.DataFrame):
        current, self.population_resolution = resolve_current_population(df)
        self.historical_df = current.copy()
        self.config = load_config()
        self.comp_config = self.config.get('compensation', {})
        self.warnings: list[str] = []
        self.df = self._valid_active_salary_population(current)
        if self.df.empty:
            raise CompensationEngineError('No active employees with a valid positive Salary are available')
        mean, _, _ = _stable_salary_stats(self.df['Salary'])
        if not np.isfinite(mean * len(self.df)):
            raise CompensationEngineError('Total payroll exceeds the finite reporting range')

    def _valid_active_salary_population(self, df: pd.DataFrame) -> pd.DataFrame:
        active = active_population(df)
        self.active_count = len(active)
        if 'Salary' not in active.columns:
            raise CompensationEngineError('Salary column is required')
        salary = pd.to_numeric(active['Salary'], errors='coerce')
        valid = salary.notna() & np.isfinite(salary) & (salary > 0)
        excluded = int((~valid).sum())
        if excluded:
            self.warnings.append(f'Excluded {excluded} active row(s) with missing or non-positive salary from compensation metrics')
        frame = active.loc[valid].copy()
        frame['Salary'] = salary.loc[valid].astype(float)
        if 'Dept' in frame.columns:
            frame['Dept'] = frame['Dept'].astype('string').str.strip().replace('', pd.NA).fillna('Unknown')
        return frame

    def calculate_salary_percentiles(self) -> pd.DataFrame:
        rows = []
        if 'Dept' not in self.df.columns:
            return pd.DataFrame()
        for dept, group in self.df.groupby('Dept'):
            salary = group['Salary']
            mean, _, _ = _stable_salary_stats(salary)
            rows.append({
                'Dept': dept, 'Headcount': len(group), 'P10': salary.quantile(.10), 'P25': salary.quantile(.25),
                'P50': salary.quantile(.50), 'P75': salary.quantile(.75), 'P90': salary.quantile(.90),
                'Mean': mean, 'Min': float(salary.min()), 'Max': float(salary.max()),
            })
        return pd.DataFrame(rows)

    def calculate_pay_equity_score(self) -> pd.DataFrame:
        if 'Dept' not in self.df.columns:
            return pd.DataFrame()
        rows = []
        for dept, group in self.df.groupby('Dept'):
            salaries = group['Salary'].dropna()
            mean, _, std = _stable_salary_stats(salaries)
            cv = float(std / mean) if mean > 0 else 0.0
            gini = _gini(salaries.to_numpy())
            score = float(np.clip(1 - ((min(cv, 1.0) + min(gini, 1.0)) / 2), 0, 1))
            status = 'Low dispersion' if score >= .8 else ('Moderate dispersion' if score >= .6 else 'High dispersion')
            rows.append({
                'Dept': dept, 'AvgSalary': mean, 'StdDev': std, 'CV': cv, 'Gini': gini,
                'EquityScore': score, 'SalaryDispersionScore': score, 'Status': status,
                'Headcount': int(len(salaries)), 'MetricSemantics': 'salary_dispersion_consistency_not_adjusted_pay_equity',
            })
        return pd.DataFrame(rows)

    def identify_salary_outliers(self, z_threshold: float = 2.5) -> pd.DataFrame:
        z_threshold = _positive_finite(z_threshold, name='z_threshold')
        if 'Dept' not in self.df.columns:
            return pd.DataFrame()
        rows = []
        for dept, group in self.df.groupby('Dept'):
            mean, _, std = _stable_salary_stats(group['Salary'])
            if std <= 0:
                continue
            scale = max(abs(mean), float(group['Salary'].abs().max()))
            normalized_salary = group['Salary'] / scale
            normalized_mean = mean / scale
            normalized_std = std / scale
            for idx, row in group.iterrows():
                z = (float(normalized_salary.loc[idx]) - normalized_mean) / normalized_std
                if abs(z) >= z_threshold:
                    deviation_pct = ((row['Salary'] / mean) - 1.0) * 100 if mean else 0.0
                    rows.append({
                        'EmployeeID': row.get('EmployeeID'), 'Dept': dept, 'Salary': row['Salary'], 'DeptAvg': mean,
                        'DeviationPct': float(deviation_pct), 'ZScore': float(z),
                        'Flag': 'Above department distribution' if z > 0 else 'Below department distribution',
                    })
        return pd.DataFrame(rows)

    def get_salary_bands(self) -> pd.DataFrame:
        salary = self.df['Salary']
        q = salary.quantile([0, .25, .5, .75, 1]).values
        labels = ['Q1 – lower quartile', 'Q2', 'Q3', 'Q4 – upper quartile']
        rows = []
        for i in range(4):
            lower, upper = float(q[i]), float(q[i + 1])
            mask = (salary >= lower) & ((salary <= upper) if i == 3 else (salary < upper))
            rows.append({'SalaryBand': labels[i], 'Lower': lower, 'Upper': upper, 'Count': int(mask.sum())})
        return pd.DataFrame(rows)

    def calculate_compa_ratio(self) -> pd.DataFrame:
        """Use supplied market-band midpoint when available; otherwise preserve compatibility fallback."""
        frame = self.df.copy()
        supplied_midpoint = 'BandMidpoint' in frame.columns
        supplied_ratio = 'CompaRatio' in frame.columns

        if supplied_midpoint:
            midpoint = pd.to_numeric(frame['BandMidpoint'], errors='coerce')
            midpoint = midpoint.where(np.isfinite(midpoint) & (midpoint > 0))
            with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
                expected_ratio = frame['Salary'] / midpoint
            expected_ratio = expected_ratio.where(np.isfinite(expected_ratio) & (expected_ratio > 0))
            ratio = expected_ratio.copy()
            semantics = 'supplied_band_midpoint_compa_ratio'

            if supplied_ratio:
                stated = pd.to_numeric(frame['CompaRatio'], errors='coerce')
                stated = stated.where(np.isfinite(stated) & (stated > 0))
                comparable = stated.notna() & expected_ratio.notna()
                consistent = pd.Series(True, index=frame.index)
                consistent.loc[comparable] = np.isclose(
                    stated.loc[comparable].to_numpy(dtype=float),
                    expected_ratio.loc[comparable].to_numpy(dtype=float),
                    rtol=1e-6,
                    atol=1e-9,
                )
                mismatched = comparable & ~consistent
                ratio = expected_ratio.where(~mismatched)
                if int(mismatched.sum()):
                    self.warnings.append(
                        f'Marked {int(mismatched.sum())} compa-ratio row(s) unavailable because supplied ratio and band midpoint were inconsistent'
                    )
                semantics = 'supplied_band_midpoint_compa_ratio_verified_against_supplied_ratio'
            frame['BandMidpoint'] = midpoint
            frame['CompaRatio'] = ratio
        elif supplied_ratio:
            ratio = pd.to_numeric(frame['CompaRatio'], errors='coerce')
            ratio = ratio.where(np.isfinite(ratio) & (ratio > 0))
            frame['CompaRatio'] = ratio
            with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
                midpoint = frame['Salary'] / ratio
            frame['BandMidpoint'] = midpoint.where(np.isfinite(midpoint))
            if frame['BandMidpoint'].isna().sum() > ratio.isna().sum():
                self.warnings.append('Some supplied compa-ratios implied non-finite band midpoints and were marked unavailable')
            semantics = 'supplied_compa_ratio'
        else:
            if 'Dept' not in frame.columns:
                return pd.DataFrame()
            midpoint = frame.groupby('Dept')['Salary'].transform('median')
            frame['BandMidpoint'] = midpoint
            frame['CompaRatio'] = frame['Salary'] / midpoint.replace(0, np.nan)
            semantics = 'relative_to_department_median_not_formal_compa_ratio'
            self.warnings.append('No external salary-band midpoint was supplied; displayed compa-ratio compatibility values are relative to department median')

        frame['CompaStatus'] = [
            'Unavailable' if pd.isna(r) or not np.isfinite(r) or r <= 0 or pd.isna(m) or not np.isfinite(m)
            else 'Below reference' if r < .8 else 'Above reference' if r > 1.2 else 'Near reference'
            for r, m in zip(frame['CompaRatio'], frame['BandMidpoint'])
        ]
        frame['MetricSemantics'] = semantics
        cols = [c for c in ['EmployeeID', 'Dept', 'Salary', 'BandMidpoint', 'CompaRatio', 'CompaStatus', 'MetricSemantics'] if c in frame.columns]
        return frame[cols]

    def correlate_salary_with_attrition(self) -> Dict[str, Any]:
        frame, _ = resolve_current_population(self.historical_df)
        if 'Attrition' not in frame.columns:
            return {'available': False, 'reason': 'Attrition unavailable'}
        salary = pd.to_numeric(frame['Salary'], errors='coerce')
        outcome = pd.to_numeric(frame['Attrition'], errors='coerce')
        valid = salary.notna() & np.isfinite(salary) & (salary > 0) & outcome.isin([0, 1])
        if valid.sum() < 20 or outcome.loc[valid].nunique() < 2:
            return {'available': False, 'reason': 'Insufficient valid salary/outcome pairs'}
        measured_salary = salary.loc[valid].astype(float)
        scale = float(measured_salary.abs().max())
        if not np.isfinite(scale) or scale <= 0:
            return {'available': False, 'reason': 'Salary has insufficient variation for correlation'}
        corr, p = stats.pointbiserialr(outcome.loc[valid].astype(int), measured_salary / scale)
        if not np.isfinite(corr) or not np.isfinite(p):
            return {'available': False, 'reason': 'Salary has insufficient variation for correlation'}
        return {
            'available': True, 'correlation': float(corr), 'p_value': float(p), 'sample_size': int(valid.sum()),
            'interpretation': 'Observed salary–attrition association; this does not establish that salary causes attrition.'
        }

    @staticmethod
    def _gender_label(series: pd.Series) -> pd.Series:
        return series.astype(str).str.strip().str.lower()

    def calculate_gender_pay_gap(self, min_group_size: int = 10) -> Dict[str, Any]:
        min_group_size = _positive_integer(min_group_size, name='min_group_size', minimum=2)
        if 'Gender' not in self.df.columns:
            return {'available': False, 'reason': 'Gender unavailable'}
        frame = self.df.copy()
        gender = self._gender_label(frame['Gender'])
        male = gender.isin({'male', 'm', 'man'})
        female = gender.isin({'female', 'f', 'woman'})
        known = male | female
        gender_known_binary_count = int(known.sum())
        gender_excluded_count = int(len(frame) - gender_known_binary_count)
        gender_coverage = float(gender_known_binary_count / len(frame)) if len(frame) else None
        male_salary, female_salary = frame.loc[male, 'Salary'], frame.loc[female, 'Salary']
        if len(male_salary) < min_group_size or len(female_salary) < min_group_size:
            return {
                'available': False,
                'reason': f'At least {min_group_size} observations per compared group are required',
                'male_n': None,
                'female_n': None,
                'groups_suppressed': True,
                'minimum_group_size': min_group_size,
                'gender_known_binary_count': gender_known_binary_count,
                'gender_excluded_count': gender_excluded_count,
                'gender_coverage': gender_coverage,
            }
        male_mean, _, _ = _stable_salary_stats(male_salary)
        female_mean, _, _ = _stable_salary_stats(female_salary)
        raw_gap = ((male_mean - female_mean) / male_mean) * 100 if male_mean else 0.0
        t_stat, p_value = _stable_welch_ttest(male_salary, female_salary)

        strata = []
        stratified_observations = 0
        if 'JobTitle' in frame.columns:
            for title, group in frame.assign(_gender=gender).groupby('JobTitle', dropna=False):
                men = group.loc[group['_gender'].isin({'male', 'm', 'man'}), 'Salary']
                women = group.loc[group['_gender'].isin({'female', 'f', 'woman'}), 'Salary']
                men_mean = _stable_salary_stats(men)[0] if len(men) else 0.0
                women_mean = _stable_salary_stats(women)[0] if len(women) else 0.0
                eligible = len(men) >= min_group_size and len(women) >= min_group_size and men_mean > 0
                title_label = 'Unknown' if pd.isna(title) else str(title)
                if eligible:
                    gap = ((men_mean - women_mean) / men_mean * 100)
                    weight = len(men) + len(women)
                    stratified_observations += weight
                    strata.append({
                        'job_title': title_label,
                        'male_n': int(len(men)), 'female_n': int(len(women)),
                        'gap_pct': float(gap), 'eligible': True, 'suppressed': False,
                        'minimum_group_size': min_group_size, 'weight': int(weight),
                    })
                else:
                    strata.append({
                        'job_title': title_label,
                        'male_n': None, 'female_n': None,
                        'gap_pct': None, 'eligible': False, 'suppressed': True,
                        'minimum_group_size': min_group_size, 'weight': 0,
                    })
        eligible_strata = [s for s in strata if s['eligible']]
        weight = sum(s['weight'] for s in eligible_strata)
        stratified = sum(s['gap_pct'] * s['weight'] for s in eligible_strata) / weight if weight else None
        inference_available = t_stat is not None and p_value is not None
        return {
            'available': True, 'raw_gap_pct': float(raw_gap), 'raw_gap_reference': 'male_mean_salary',
            'male_n': len(male_salary), 'female_n': len(female_salary),
            'gender_known_binary_count': gender_known_binary_count,
            'gender_excluded_count': gender_excluded_count,
            'gender_coverage': gender_coverage,
            'welch_t_stat': t_stat,
            'p_value': p_value,
            'is_significant': bool(inference_available and p_value < .05),
            'inference_available': inference_available,
            'job_title_stratified_gap_pct': float(stratified) if stratified is not None else None,
            'eligible_job_title_strata': len(eligible_strata), 'job_title_strata': strata,
            'job_title_stratified_observations': int(stratified_observations),
            'job_title_stratified_excluded_count': int(gender_known_binary_count - stratified_observations),
            'semantics': 'descriptive_and_job_title_stratified_gap_not_regression_adjusted_equity',
            'warning': 'Pay-gap statistics are disparity indicators, not a legal or causal determination of pay equity.'
        }

    def get_salary_by_tenure(self) -> pd.DataFrame:
        if 'Tenure' not in self.df.columns:
            return pd.DataFrame()
        frame = self.df.copy()
        tenure = pd.to_numeric(frame['Tenure'], errors='coerce')
        tenure = tenure.where(np.isfinite(tenure) & (tenure >= 0))
        frame['TenureBucket'] = pd.cut(tenure, bins=[0, 1, 2, 5, 10, float('inf')], labels=['<1 year', '1-2 years', '2-5 years', '5-10 years', '10+ years'], right=False).cat.add_categories('Unknown').fillna('Unknown')
        rows = []
        for bucket, group in frame.groupby('TenureBucket', observed=False):
            salary = group['Salary']
            if salary.empty:
                rows.append({'TenureBucket': bucket, 'Mean': np.nan, 'Median': np.nan, 'Min': np.nan, 'Max': np.nan, 'Count': 0})
                continue
            mean, median, _ = _stable_salary_stats(salary)
            rows.append({'TenureBucket': bucket, 'Mean': mean, 'Median': median, 'Min': float(salary.min()), 'Max': float(salary.max()), 'Count': int(len(salary))})
        return pd.DataFrame(rows)

    def get_compensation_summary(self) -> Dict[str, Any]:
        salary = self.df['Salary']
        mean, median, deviation = _stable_salary_stats(salary)
        return {
            'total_payroll': float(mean * len(salary)), 'avg_salary': mean, 'median_salary': median,
            'min_salary': float(salary.min()), 'max_salary': float(salary.max()), 'salary_range': float(salary.max() - salary.min()),
            'std_dev': deviation, 'headcount': int(len(salary)),
            'active_count': int(self.active_count),
            'salary_observations': int(len(salary)),
            'excluded_salary_count': int(self.active_count - len(salary)),
            'salary_coverage': float(len(salary) / self.active_count) if self.active_count else None,
            'population': 'current_active_employees_with_valid_positive_salary',
        }

    def _aggregate_outlier_summary(self) -> Dict[str, Any]:
        outliers = self.identify_salary_outliers()
        if outliers.empty:
            return {
                'count': 0, 'departments_affected': 0,
                'above_count': 0, 'below_count': 0,
                'population': 'current_active_employees_with_valid_positive_salary',
                'semantics': 'aggregate_salary_distribution_outlier_count_not_employee_ranking',
            }
        flags = outliers['Flag'].astype(str)
        return {
            'count': int(len(outliers)),
            'departments_affected': int(outliers['Dept'].nunique()) if 'Dept' in outliers else 0,
            'above_count': int(flags.str.startswith('Above').sum()),
            'below_count': int(flags.str.startswith('Below').sum()),
            'population': 'current_active_employees_with_valid_positive_salary',
            'semantics': 'aggregate_salary_distribution_outlier_count_not_employee_ranking',
        }

    def analyze_all(self) -> Dict[str, Any]:
        return {
            'summary': self.get_compensation_summary(),
            'equity': self.calculate_pay_equity_score(),
            'salary_dispersion': self.calculate_pay_equity_score(),
            'outliers': self._aggregate_outlier_summary(),
            'gender_pay_gap': self.calculate_gender_pay_gap(),
            'salary_attrition_association': self.correlate_salary_with_attrition(),
            'warnings': list(dict.fromkeys(self.warnings)),
        }
