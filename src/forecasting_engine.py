"""Observed monthly workforce extrapolation with a naive validation baseline.

Missing periods remain missing. No invented daily observations or confidence
bands are used. Forecast errors are retrospective validation, not guarantees.
"""
from typing import Any
import numpy as np
import pandas as pd
from src.population import active_population


class ForecastingEngine:
    def __init__(self, history_df: pd.DataFrame):
        self.history_df = history_df.copy()

    def forecast_metric(self, metric: str, periods: int = 12, freq: str = 'M') -> dict[str, Any]:
        if not 1 <= periods <= 36:
            return {'success': False, 'reason': 'Forecast horizon must be 1–36 monthly periods'}
        if freq not in ('M', 'ME'):
            return {'success': False, 'reason': 'Only monthly workforce forecasts are supported'}
        if metric not in ('headcount', 'salary', 'Salary', 'avg_salary'):
            return {'success': False, 'reason': 'Only active headcount and mean salary are supported; turnover requires dated exit events and exposure denominators.'}
        date_col = 'SnapshotDate' if 'SnapshotDate' in self.history_df else 'snapshot_date'
        if date_col not in self.history_df:
            return {'success': False, 'reason': 'Observed snapshot dates are required'}
        frame = self.history_df.copy()
        frame[date_col] = pd.to_datetime(frame[date_col], errors='coerce', utc=True)
        frame = frame.dropna(subset=[date_col])
        observations = {}
        for date, group in frame.groupby(date_col, sort=True):
            current = active_population(group)
            if metric == 'headcount':
                value = float(len(current))
            elif 'Salary' in current:
                values = pd.to_numeric(current['Salary'], errors='coerce')
                values = values[np.isfinite(values) & (values > 0)]
                value = float(values.mean()) if len(values) else np.nan
            else:
                return {'success': False, 'reason': 'Salary is unavailable'}
            # Last observed census in each month, never sum repeated snapshots.
            observations[date.tz_localize(None).to_period('M')] = value
        series = pd.Series(observations, dtype=float).sort_index()
        if len(series) < 12:
            return {'success': False, 'reason': 'At least 12 observed monthly censuses are required'}
        expected = pd.period_range(series.index.min(), series.index.max(), freq='M')
        if not series.index.equals(expected) or not np.isfinite(series).all():
            return {'success': False, 'reason': 'Missing months or invalid observations; supply complete monthly censuses rather than interpolating history'}
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        train, validation = series.iloc[:-3], series.iloc[-3:]
        trend = 'add' if len(train) >= 12 else None
        def fit(values):
            return ExponentialSmoothing(values.to_numpy(), trend=trend,
                                        initialization_method='estimated').fit()
        naive_mae = float(np.abs(validation.to_numpy() - train.iloc[-1]).mean())
        try:
            predicted = np.maximum(0, fit(train).forecast(3))
            model_mae = float(np.abs(validation.to_numpy() - predicted).mean())
            use_model = np.isfinite(model_mae) and model_mae < naive_mae
            forecast = np.maximum(0, fit(series).forecast(periods)) if use_model else np.repeat(series.iloc[-1], periods)
        except (ValueError, FloatingPointError):
            model_mae, use_model = None, False
            forecast = np.repeat(series.iloc[-1], periods)
        dates = pd.period_range(series.index[-1] + 1, periods=periods, freq='M')
        return {
            'success': True, 'metric': metric,
            'history': [{'date': d.to_timestamp(how='end').strftime('%Y-%m-%d'), 'value': float(v)} for d, v in series.items()],
            'forecast': [{'date': d.to_timestamp(how='end').strftime('%Y-%m-%d'), 'value': float(v)} for d, v in zip(dates, forecast)],
            'model': 'exponential_smoothing' if use_model else 'last_observation_baseline',
            'validation': {'months': 3, 'model_mae': model_mae, 'naive_mae': naive_mae,
                           'scope': 'last_three_observed_months_used_for_model_selection'},
            'uncertainty': 'Prediction intervals have not been estimated.',
            'semantics': 'monthly_active_workforce_extrapolation_not_causal_forecast',
            'assumptions': ['Each source date must be a complete workforce census; partial exports invalidate comparisons.'],
        }
