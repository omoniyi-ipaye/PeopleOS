"""Observed monthly workforce extrapolation with a naive validation baseline.

Missing periods remain missing. No invented daily observations or confidence
bands are used. Forecast errors are retrospective validation, not guarantees.
"""
from __future__ import annotations

from typing import Any
import warnings

import numpy as np
import pandas as pd

from src.population import active_population


class ForecastingEngine:
    def __init__(self, history_df: pd.DataFrame):
        self.history_df = history_df.copy(deep=True)

    @staticmethod
    def _source_timestamp(value: Any) -> pd.Timestamp:
        """Parse a source census timestamp while preserving its calendar date.

        SnapshotDate is a census label, not an instant to be rebucketed into UTC.
        Removing timezone information without conversion preserves the month that
        the source system explicitly assigned to the census.
        """
        try:
            value = pd.Timestamp(value)
        except (TypeError, ValueError, OverflowError):
            return pd.NaT
        if pd.isna(value):
            return pd.NaT
        if value.tzinfo is not None:
            value = value.tz_localize(None)
        return value

    def forecast_metric(self, metric: str, periods: int = 12, freq: str = 'M') -> dict[str, Any]:
        if not isinstance(periods, int) or isinstance(periods, bool) or not 1 <= periods <= 36:
            return {'success': False, 'reason': 'Forecast horizon must be 1–36 monthly periods'}
        if freq not in ('M', 'ME'):
            return {'success': False, 'reason': 'Only monthly workforce forecasts are supported'}
        if metric not in ('headcount', 'salary', 'Salary', 'avg_salary'):
            return {'success': False, 'reason': 'Only active headcount and mean salary are supported; turnover requires dated exit events and exposure denominators.'}
        if metric != 'headcount':
            from src.platform.runtime_loader import pay_basis_is_confirmed
            if not pay_basis_is_confirmed(self.history_df):
                return {'success': False, 'reason': 'Salary forecasts require explicit annual pay and one shared currency across all historical snapshots.'}

        date_col = 'SnapshotDate' if 'SnapshotDate' in self.history_df else 'snapshot_date'
        if date_col not in self.history_df:
            return {'success': False, 'reason': 'Observed snapshot dates are required'}

        frame = self.history_df.copy(deep=True)
        parsed = frame[date_col].map(self._source_timestamp)
        invalid_dates = int(parsed.isna().sum())
        if invalid_dates:
            return {
                'success': False,
                'reason': f'{invalid_dates} workforce rows have missing or invalid snapshot dates; every row in a historical census must have a valid date.',
            }
        frame['_source_snapshot'] = parsed
        frame['_source_month'] = parsed.map(lambda value: value.to_period('M'))

        observations: dict[pd.Period, float] = {}
        observation_coverage: dict[pd.Period, dict[str, int]] = {}
        try:
            for month, month_frame in frame.groupby('_source_month', sort=True):
                latest = month_frame['_source_snapshot'].max()
                census = month_frame[month_frame['_source_snapshot'] == latest].drop(columns=['_source_snapshot', '_source_month'])
                current = active_population(census)
                if metric == 'headcount':
                    value = float(len(current))
                    measured = len(current)
                elif 'Salary' in current:
                    values = pd.to_numeric(current['Salary'], errors='coerce')
                    valid = values.notna() & np.isfinite(values) & (values > 0)
                    measured = int(valid.sum())
                    if measured != len(current):
                        return {
                            'success': False,
                            'reason': (
                                f'Salary coverage is incomplete for {month}: {measured} of {len(current)} active employees '
                                'have a valid positive finite salary. Forecasting will not compare shifting measurement populations.'
                            ),
                        }
                    value = float(values[valid].mean()) if measured else np.nan
                else:
                    return {'success': False, 'reason': 'Salary is unavailable'}
                observations[month] = value
                observation_coverage[month] = {'active_population': int(len(current)), 'measured_population': int(measured)}
        except ValueError as exc:
            return {'success': False, 'reason': f'Historical census is ambiguous: {exc}'}

        series = pd.Series(observations, dtype=float).sort_index()
        if len(series) < 12:
            return {'success': False, 'reason': 'At least 12 observed monthly censuses are required'}
        expected = pd.period_range(series.index.min(), series.index.max(), freq='M')
        if not series.index.equals(expected) or not np.isfinite(series).all():
            return {'success': False, 'reason': 'Missing months or invalid observations; supply complete monthly censuses rather than interpolating history'}

        train, validation = series.iloc[:-3], series.iloc[-3:]
        naive_mae = float(np.abs(validation.to_numpy() - train.iloc[-1]).mean())
        model_mae: float | None = None
        use_model = False

        # A constant history is exactly represented by the last-observation
        # baseline; fitting Holt-Winters adds no information and can emit divide-
        # by-zero diagnostics from a zero residual sum of squares.
        if series.nunique(dropna=False) > 1:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            trend = 'add' if len(train) >= 12 else None

            def fit(values: pd.Series):
                return ExponentialSmoothing(
                    values.to_numpy(dtype=float), trend=trend, initialization_method='estimated'
                ).fit()

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('error', RuntimeWarning)
                    predicted = np.asarray(fit(train).forecast(3), dtype=float)
                    if not np.isfinite(predicted).all():
                        raise FloatingPointError('non-finite validation forecast')
                    predicted = np.maximum(0, predicted)
                    candidate_mae = float(np.abs(validation.to_numpy(dtype=float) - predicted).mean())
                    if np.isfinite(candidate_mae):
                        model_mae = candidate_mae
                        use_model = candidate_mae < naive_mae
                    if use_model:
                        forecast = np.asarray(fit(series).forecast(periods), dtype=float)
                        if not np.isfinite(forecast).all():
                            raise FloatingPointError('non-finite forecast')
                        forecast = np.maximum(0, forecast)
                    else:
                        forecast = np.repeat(float(series.iloc[-1]), periods)
            except (ValueError, FloatingPointError, RuntimeWarning, OverflowError):
                model_mae, use_model = None, False
                forecast = np.repeat(float(series.iloc[-1]), periods)
        else:
            forecast = np.repeat(float(series.iloc[-1]), periods)

        if not np.isfinite(forecast).all():
            return {'success': False, 'reason': 'Forecast calculation produced a non-finite result and was withheld.'}

        dates = pd.period_range(series.index[-1] + 1, periods=periods, freq='M')
        return {
            'success': True,
            'metric': metric,
            'history': [
                {
                    'date': d.to_timestamp(how='end').strftime('%Y-%m-%d'),
                    'value': float(v),
                    'active_population': observation_coverage[d]['active_population'],
                    'measured_population': observation_coverage[d]['measured_population'],
                }
                for d, v in series.items()
            ],
            'forecast': [{'date': d.to_timestamp(how='end').strftime('%Y-%m-%d'), 'value': float(v)} for d, v in zip(dates, forecast)],
            'model': 'exponential_smoothing' if use_model else 'last_observation_baseline',
            'validation': {
                'months': 3,
                'model_mae': model_mae,
                'naive_mae': naive_mae,
                'scope': 'last_three_observed_months_used_for_model_selection',
            },
            'uncertainty': 'Prediction intervals have not been estimated.',
            'semantics': 'monthly_active_workforce_extrapolation_not_causal_forecast',
            'assumptions': ['Each source date must be a complete workforce census; partial exports invalidate comparisons.'],
        }
