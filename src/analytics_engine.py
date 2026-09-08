"""Deterministic descriptive analytics for PeopleOS.

Metric semantics are explicit:
- headcount = current active employees;
- record_count = current employee observations with known/unknown status;
- observed_attrition_share = share of current rows observed as departed.

`turnover_rate`/`Turnover_Rate` remain compatibility aliases only. They do not
represent a period turnover rate because the canonical employee dataset does not
carry the exposure denominator required for that calculation.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats

from src.logger import get_logger
from src.population import active_population, observed_attrition_share, resolve_current_population
from src.utils import load_config

logger = get_logger('analytics_engine')


def _valid_numeric(series: pd.Series, name: str) -> pd.Series:
    values = pd.to_numeric(series, errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
    if name == 'Salary':
        values = values[values > 0]
    elif name == 'Tenure':
        values = values[values >= 0]
    elif name == 'Age':
        values = values[values.between(1, 120)]
    elif name == 'LastRating':
        values = values[values.between(1, 5)]
    return values


class AnalyticsEngine:
    def __init__(self, df: pd.DataFrame):
        self.df, self.population_resolution = resolve_current_population(df)
        self.active_df = active_population(self.df)
        self.config = load_config()
        self.analytics_config = self.config.get('analytics', {})
        self.high_risk_threshold = self.analytics_config.get('high_risk_dept_threshold', 0.20)
        self._validate_data()

    def _validate_data(self) -> None:
        required = ['EmployeeID', 'Dept', 'Tenure', 'Salary', 'LastRating', 'Age']
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            logger.warning('Missing columns for analytics: %s', missing)

    def get_headcount(self) -> int:
        """Current active employee count."""
        return len(self.active_df)

    def get_record_count(self) -> int:
        return len(self.df)

    def get_observed_attrition_share(self) -> Optional[float]:
        return observed_attrition_share(self.df)

    def get_turnover_rate(self) -> Optional[float]:
        """Deprecated compatibility alias for observed attrition share."""
        return self.get_observed_attrition_share()

    def get_department_aggregates(self) -> pd.DataFrame:
        if 'Dept' not in self.df.columns:
            return pd.DataFrame()

        rows: list[dict[str, Any]] = []
        for dept, current in self.df.groupby('Dept', dropna=False):
            active = current[current['Attrition'] == 0] if 'Attrition' in current.columns else current
            known_attrition = current['Attrition'].dropna() if 'Attrition' in current.columns else pd.Series(dtype=float)
            attrition_share = float(known_attrition.mean()) if not known_attrition.empty else None
            row: dict[str, Any] = {
                'Dept': str(dept) if pd.notna(dept) else 'Unknown',
                'Total_Records': int(len(current)),
                'Outcome_Observations': int(len(known_attrition)),
                'Headcount': int(len(active)),
                'Observed_Attrition_Share': attrition_share,
                # Compatibility alias; not a period turnover rate.
                'Turnover_Rate': attrition_share,
            }
            if 'Salary' in active.columns:
                valid = _valid_numeric(active['Salary'], 'Salary')
                row['Avg_Salary'] = float(valid.mean()) if not valid.empty else None
                row['Median_Salary'] = float(valid.median()) if not valid.empty else None
            for source, output in [('Tenure', 'Avg_Tenure'), ('LastRating', 'Avg_Rating'), ('Age', 'Avg_Age')]:
                if source in active.columns:
                    values = _valid_numeric(active[source], source)
                    row[output] = float(values.mean()) if not values.empty else None
            rows.append(row)
        return pd.DataFrame(rows)

    def get_correlations(self, target_column: str = 'Attrition', max_features: int = 20) -> pd.DataFrame:
        if target_column not in self.df.columns:
            return pd.DataFrame()
        numeric = self.df.select_dtypes(include=[np.number]).drop(columns=['EmployeeID'], errors='ignore').replace([np.inf, -np.inf], np.nan)
        for col in numeric:
            numeric[col] = _valid_numeric(numeric[col], col).reindex(numeric.index)
        if target_column not in numeric.columns or numeric[target_column].dropna().nunique() < 2:
            return pd.DataFrame()
        if len(numeric.columns) > max_features + 1:
            variance = numeric.var(numeric_only=True).sort_values(ascending=False)
            keep = [target_column] + [c for c in variance.index if c != target_column][:max_features]
            numeric = numeric[[c for c in keep if c in numeric.columns]]
        corr = numeric.corr(numeric_only=True)[target_column].drop(target_column, errors='ignore').dropna()
        return pd.DataFrame({'Feature': corr.index, 'Correlation': corr.values, 'Abs_Correlation': corr.abs().values}).sort_values('Abs_Correlation', ascending=False)

    def get_summary_statistics(self) -> dict:
        attrition_share = self.get_observed_attrition_share()
        result: dict[str, Any] = {
            'headcount': self.get_headcount(),
            'record_count': self.get_record_count(),
            'active_count': self.get_headcount(),
            'observed_attrition_share': attrition_share,
            'turnover_rate': attrition_share,  # compatibility only
            'turnover_rate_semantics': 'observed_attrition_share_not_period_turnover',
            'department_count': int(self.active_df['Dept'].nunique()) if 'Dept' in self.active_df.columns else 0,
            'population_as_of_date': self.population_resolution.as_of_date,
            'snapshot_history': self.population_resolution.snapshot_history,
        }
        for col in ('Salary', 'Tenure', 'LastRating', 'Age'):
            if col in self.active_df.columns:
                values = _valid_numeric(self.active_df[col], col)
                result[f'{col.lower()}_mean'] = float(values.mean()) if not values.empty else None
                result[f'{col.lower()}_median'] = float(values.median()) if not values.empty else None
                result[f'{col.lower()}_std'] = float(values.std()) if len(values) > 1 else None
        if 'Attrition' in self.df.columns:
            result['attrition_count'] = int((self.df['Attrition'] == 1).sum())
            result['attrition_known_count'] = int(self.df['Attrition'].notna().sum())
        temporal = self.get_temporal_stats(active_only=True)
        if temporal:
            result['temporal'] = temporal
        return result

    def get_temporal_stats(self, active_only: bool = False) -> dict:
        frame = self.active_df if active_only else self.df
        result = {}
        for col, key in [('RatingVelocity', 'avg_velocity'), ('PromotionLag', 'avg_promo_lag'), ('SalaryGrowth', 'avg_salary_growth')]:
            if col in frame.columns:
                values = _valid_numeric(frame[col], col)
                if not values.empty:
                    result[key] = float(values.mean())
        return result

    def get_tenure_distribution(self) -> pd.DataFrame:
        if 'Tenure' not in self.active_df.columns:
            return pd.DataFrame()
        active = self.active_df.copy()
        bins = [0, 1, 2, 5, 10, float('inf')]
        labels = ['<1 year', '1-2 years', '2-5 years', '5-10 years', '10+ years']
        active['Tenure_Bucket'] = pd.cut(_valid_numeric(active['Tenure'], 'Tenure').reindex(active.index), bins=bins, labels=labels, right=False).cat.add_categories('Unknown').fillna('Unknown')
        distribution = active['Tenure_Bucket'].value_counts(sort=False).rename_axis('Tenure_Range').reset_index(name='Count')
        # Attrition outcome by tenure is calculated over current records, because active-only data cannot contain departed outcomes.
        if 'Attrition' in self.df.columns:
            current = self.df.copy()
            current['Tenure_Bucket'] = pd.cut(_valid_numeric(current['Tenure'], 'Tenure').reindex(current.index), bins=bins, labels=labels, right=False).cat.add_categories('Unknown').fillna('Unknown')
            shares = current.groupby('Tenure_Bucket', observed=False)['Attrition'].mean().rename('Observed_Attrition_Share')
            distribution = distribution.merge(shares.reset_index().rename(columns={'Tenure_Bucket': 'Tenure_Range'}), on='Tenure_Range', how='left')
            distribution['Turnover_Rate'] = distribution['Observed_Attrition_Share']
        return distribution

    def get_age_distribution(self) -> pd.DataFrame:
        if 'Age' not in self.active_df.columns:
            return pd.DataFrame()
        bins = [0, 25, 35, 45, 55, float('inf')]
        labels = ['Under 25', '25-34', '35-44', '45-54', '55+']
        bucket = pd.cut(_valid_numeric(self.active_df['Age'], 'Age').reindex(self.active_df.index), bins=bins, labels=labels, right=False).cat.add_categories('Unknown').fillna('Unknown')
        return bucket.value_counts(sort=False).rename_axis('Age_Range').reset_index(name='Count')

    def get_salary_bands(self) -> pd.DataFrame:
        if 'Salary' not in self.active_df.columns:
            return pd.DataFrame()
        salary = _valid_numeric(self.active_df['Salary'], 'Salary')
        if salary.empty:
            return pd.DataFrame()
        quantiles = salary.quantile([0, .25, .5, .75, 1]).values
        rows = []
        for i in range(4):
            lower, upper = float(quantiles[i]), float(quantiles[i + 1])
            include_upper = i == 3
            mask = (salary >= lower) & ((salary <= upper) if include_upper else (salary < upper))
            rows.append({'Band': f'Q{i + 1}', 'Lower': lower, 'Upper': upper, 'Count': int(mask.sum())})
        return pd.DataFrame(rows)

    def get_high_risk_departments(self, threshold: Optional[float] = None) -> pd.DataFrame:
        threshold = self.high_risk_threshold if threshold is None else threshold
        stats_df = self.get_department_aggregates()
        metric = 'Observed_Attrition_Share'
        if metric not in stats_df.columns:
            return pd.DataFrame()
        return stats_df[stats_df[metric].fillna(-1) > threshold].sort_values(metric, ascending=False)

    def compare_groups(self, group_col: str, metric_col: str) -> dict:
        if group_col not in self.active_df.columns or metric_col not in self.active_df.columns:
            return {'success': False, 'reason': 'Columns not found'}
        frame = self.active_df.dropna(subset=[group_col, metric_col]).copy()
        frame[metric_col] = pd.to_numeric(frame[metric_col], errors='coerce').replace([np.inf, -np.inf], np.nan)
        if metric_col == 'Salary':
            frame = frame[frame[metric_col] > 0]
        grouped = frame.dropna(subset=[metric_col]).groupby(group_col)[metric_col]
        eligible = [(name, group.values) for name, group in grouped if len(group) > 5]
        if len(eligible) < 2:
            return {'success': False, 'reason': 'Not enough groups with data (>5 samples)'}
        names = [name for name, _ in eligible]
        values = [vals for _, vals in eligible]
        try:
            if len(values) == 2:
                stat, p_value = stats.ttest_ind(values[0], values[1], equal_var=False, nan_policy='omit')
                test_name = "Welch's T-Test"
            else:
                stat, p_value = stats.f_oneway(*values)
                test_name = 'One-way ANOVA'
            if not np.isfinite(stat) or not np.isfinite(p_value):
                return {'success': False, 'reason': 'Group variation is insufficient for a finite statistical test.'}
            return {
                'success': True, 'test_name': test_name, 'statistic': float(stat), 'p_value': float(p_value),
                'is_significant': bool(p_value < .05), 'groups_compared': names,
                'interpretation': f"Observed group difference for {metric_col}; {test_name} p={p_value:.4f}. Statistical significance does not establish causation or unfairness."
            }
        except Exception as exc:
            logger.error('Statistical test failed: %s', exc)
            return {'success': False, 'reason': str(exc)}

    def get_confidence_interval(self, col: str, confidence: float = 0.95) -> Optional[tuple]:
        if col not in self.active_df.columns:
            return None
        if not 0 < confidence < 1:
            raise ValueError('confidence must be between zero and one')
        data = _valid_numeric(self.active_df[col], col)
        if len(data) < 2:
            return None
        mean = data.mean()
        sem = stats.sem(data)
        margin = sem * stats.t.ppf((1 + confidence) / 2, len(data) - 1)
        return float(mean - margin), float(mean + margin)
