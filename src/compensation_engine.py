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
    scale = float(np.max(np.abs(array)))
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
            rows.append({
                'Dept': dept, 'Headcount': len(group), 'P10': salary.quantile(.10), 'P25': salary.quantile(.25),
                'P50': salary.quantile(.50), 'P75': salary.quantile(.75), 'P90': salary.quantile(.90),
                'Mean': salary.mean(), 'Min': salary.min(), 'Max': salary.max(),
            })
        return pd.DataFrame(rows)

    def calculate_pay_equity_score(self) -> pd.DataFrame:
        """Compatibility endpoint for within-department salary dispersion.

        `EquityScore` is NOT an adjusted pay-equity conclusion. It is retained so
        existing API clients continue to work while the semantic field is exposed
        as `MetricSemantics=salary_dispersion_consistency`.
        """
        if 'Dept' not in self.df.columns:
            return pd.DataFrame()
        rows = []
        for dept, group in self.df.groupby('Dept'):
            salaries = group['Salary'].dropna()
            mean = float(salaries.mean()) if not salaries.empty else 0.0
            std = float(salaries.std(ddof=1)) if len(salaries) > 1 else 0.0
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
            mean = group['Salary'].mean()
            std = group['Salary'].std(ddof=1)
            if not np.isfinite(std) or std <= 0:
                continue
            for _, row in group.iterrows():
                z = (row['Salary'] - mean) / std
                if abs(z) >= z_threshold:
                    rows.append({
                        'EmployeeID': row.get('EmployeeID'), 'Dept': dept, 'Salary': row['Salary'], 'DeptAvg': mean,
                        'DeviationPct': ((row['Salary'] - mean) / mean) * 100 if mean else 0,
                        'ZScore': float(z), 'Flag': 'Above department distribution' if z > 0 else 'Below department distribution',
                    })
        return pd.DataFrame(rows)

    def get_salary_bands(self) -> pd.DataFrame:
        """Return salary quartiles; quartiles are not job/career levels."""
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
        """Calculate true compa-ratio only when a supplied ratio/band basis exists.

        If no external band midpoint exists in the schema, return a clearly named
        relative-to-department-median ratio in the compatibility columns.
        """
        frame = self.df.copy()
        if 'CompaRatio' in frame.columns:
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
        """Association across current employee records, including departed outcomes."""
        frame, _ = resolve_current_population(self.historical_df)
        if 'Attrition' not in frame.columns:
            return {'available': False, 'reason': 'Attrition unavailable'}
        salary = pd.to_numeric(frame['Salary'], errors='coerce')
        outcome = pd.to_numeric(frame['Attrition'], errors='coerce')
        valid = salary.notna() & np.isfinite(salary) & (salary > 0) & outcome.isin([0, 1])
        if valid.sum() < 20 or outcome.loc[valid].nunique() < 2:
            return {'available': False, 'reason': 'Insufficient valid salary/outcome pairs'}
        corr, p = stats.pointbiserialr(outcome.loc[valid].astype(int), salary.loc[valid].astype(float))
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
        male_salary, female_salary = frame.loc[male, 'Salary'], frame.loc[female, 'Salary']
        if len(male_salary) < min_group_size or len(female_salary) < min_group_size:
            return {'available': False, 'reason': f'At least {min_group_size} observations per compared group are required'}
        male_mean = float(male_salary.mean())
        female_mean = float(female_salary.mean())
        raw_gap = (male_mean - female_mean) / male_mean * 100 if male_mean else 0.0
        t_stat, p_value = stats.ttest_ind(male_salary, female_salary, equal_var=False, nan_policy='omit')

        strata = []
        if 'JobTitle' in frame.columns:
            for title, group in frame.assign(_gender=gender).groupby('JobTitle'):
                men = group.loc[group['_gender'].isin({'male', 'm', 'man'}), 'Salary']
                women = group.loc[group['_gender'].isin({'female', 'f', 'woman'}), 'Salary']
                men_mean = float(men.mean()) if len(men) else 0.0
                women_mean = float(women.mean()) if len(women) else 0.0
                eligible = len(men) >= min_group_size and len(women) >= min_group_size and men_mean > 0
                gap = ((men_mean - women_mean) / men_mean * 100) if eligible else None
                strata.append({'job_title': title, 'male_n': len(men), 'female_n': len(women), 'gap_pct': gap, 'eligible': eligible, 'weight': len(men) + len(women) if eligible else 0})
        eligible = [s for s in strata if s['eligible']]
        weight = sum(s['weight'] for s in eligible)
        stratified = sum(s['gap_pct'] * s['weight'] for s in eligible) / weight if weight else None
        return {
            'available': True, 'raw_gap_pct': float(raw_gap), 'male_n': len(male_salary), 'female_n': len(female_salary),
            'welch_t_stat': float(t_stat) if np.isfinite(t_stat) else None,
            'p_value': float(p_value) if np.isfinite(t_stat) and np.isfinite(p_value) else None,
            'is_significant': bool(np.isfinite(t_stat) and np.isfinite(p_value) and p_value < .05),
            'inference_available': bool(np.isfinite(t_stat) and np.isfinite(p_value)),
            'job_title_stratified_gap_pct': float(stratified) if stratified is not None else None,
            'eligible_job_title_strata': len(eligible), 'job_title_strata': strata,
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
        grouped = frame.groupby('TenureBucket', observed=False)['Salary'].agg(['mean', 'median', 'min', 'max', 'count']).reset_index()
        return grouped.rename(columns={'mean': 'Mean', 'median': 'Median', 'min': 'Min', 'max': 'Max', 'count': 'Count'})

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

    def analyze_all(self) -> Dict[str, Any]:
        return {
            'summary': self.get_compensation_summary(),
            'equity': self.calculate_pay_equity_score(),
            'salary_dispersion': self.calculate_pay_equity_score(),
            'outliers': self.identify_salary_outliers(),
            'gender_pay_gap': self.calculate_gender_pay_gap(),
            'salary_attrition_association': self.correlate_salary_with_attrition(),
            'warnings': list(dict.fromkeys(self.warnings)),
        }
