"""
Forecasting Engine module for PeopleOS.

Provides time-series forecasting for key HR metrics (Headcount, Turnover, Salary)
using Exponential Smoothing (Holt-Winters).
"""

from typing import Dict, Any
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from src.logger import get_logger

logger = get_logger('forecasting_engine')

class ForecastingEngine:
    """
    Engine for predicting future HR trends using Time Series Analysis.
    """
    
    def __init__(self, history_df: pd.DataFrame):
        """
        Initialize with historical data.
        
        Args:
            history_df: DataFrame containing historical snapshots.
                        Must have 'snapshot_date' and metric columns.
        """
        self.history_df = history_df.copy()
        
        # Ensure date format
        if 'snapshot_date' in self.history_df.columns:
            self.history_df['snapshot_date'] = pd.to_datetime(self.history_df['snapshot_date'])
        
    def forecast_metric(self, metric: str, periods: int = 12, freq: str = 'M') -> Dict[str, Any]:
        """
        Forecast a specific metric for N periods into the future.
        
        Args:
            metric: Column name to forecast (e.g., 'headcount', 'avg_salary').
            periods: Number of periods to predict.
            freq: Frequency ('M' for Monthly).
            
        Returns:
            Dictionary containing historical data and forecast points.
        """
        try:
            # 1. Prepare Time Series
            if metric == 'headcount':
                # Aggregation for headcount
                ts = self.history_df.groupby(pd.Grouper(key='snapshot_date', freq=freq))['EmployeeID'].nunique()
            elif metric == 'turnover_rate':
                # Aggregation for turnover (requires attrition calculation per month)
                # This is complex; simplified: count attrition events / headcount
                # For now, let's assume pre-calculated monthly metrics passed in history_df
                # Or we calculate it here based on 'Attrition' events in snapshots?
                # BETTER: Use a helper to aggregate snapshots first.
                return self._forecast_turnover(periods, freq)
            else:
                # Average of numeric column (e.g., Salary)
                ts = self.history_df.groupby(pd.Grouper(key='snapshot_date', freq=freq))[metric].mean()
            
            # Fill missing months
            ts = ts.asfreq(freq).ffill().bfill()
            
            if len(ts) < 6:
                logger.warning(f"Insufficient history for {metric} forecasting (need >6 points).")
                return {'success': False, 'reason': 'Insufficient history'}

            # 2. Train Model (Exponential Smoothing)
            # Use additive trend and seasonality if enough data
            trend = 'add' if len(ts) >= 12 else None
            seasonal = 'add' if len(ts) >= 24 else None
            seasonal_periods = 12 if seasonal else None
            
            model = ExponentialSmoothing(
                ts, 
                trend=trend, 
                seasonal=seasonal,
                seasonal_periods=seasonal_periods,
                initialization_method="estimated"
            )
            fit = model.fit()
            
            # 3. Forecast
            forecast = fit.forecast(periods)
            
            # 4. Format Results
            history_points = [{'date': d.strftime('%Y-%m-%d'), 'value': float(v)} for d, v in ts.items()]
            forecast_points = [{'date': d.strftime('%Y-%m-%d'), 'value': float(v)} for d, v in forecast.items()]
            
            # Add simple confidence intervals (heuristic for ES as statsmodels CI is complex)
            # +/- 10% growing over time
            for i, point in enumerate(forecast_points):
                margin = 0.05 + (0.01 * i) # 5% start, growing 1% per month
                val = point['value']
                point['lower'] = val * (1 - margin)
                point['upper'] = val * (1 + margin)
            
            return {
                'success': True,
                'metric': metric,
                'history': history_points,
                'forecast': forecast_points,
                'model_params': fit.params_formatted.to_dict()
            }
            
        except Exception as e:
            logger.error(f"Forecasting failed for {metric}: {e}")
            return {'success': False, 'reason': str(e)}

    def _forecast_turnover(self, periods: int, freq: str) -> Dict[str, Any]:
        """Special handling for turnover rate forecasting."""
         # Calculate monthly turnover rate
        monthly_groups = self.history_df.groupby(pd.Grouper(key='snapshot_date', freq=freq))
        
        turnover_rates = []
        dates = []
        
        for date, group in monthly_groups:
            # Count attrition events in this snapshot (assuming 'Attrition' means left in this period)
            # Note: With snapshots, Attrition=1 usually means "Left company". 
            # We need to count NEW leavers only. 
            # Simplified: If we have monthly snapshots, we check Attrition=1 count / Total count
            
            # Ideally, we should look at transitions between snapshots.
            # But for "Credibility", let's use the explicit 'Attrition' flag counts if available
            headcount = group['EmployeeID'].nunique()
            leavers = group['Attrition'].sum() if 'Attrition' in group.columns else 0
            rate = (leavers / headcount) if headcount > 0 else 0
            
            turnover_rates.append(rate)
            dates.append(date)
            
        ts = pd.Series(turnover_rates, index=dates).asfreq(freq).fillna(0)
        
        # Just use the generic forecast logic on this prepared data
        # We allow recursion but avoid converting to Series again
        
        # Copy-paste logic from above to avoid recursion complexity with TS input
        try:
             model = ExponentialSmoothing(ts, trend=None, seasonal=None, initialization_method="simple")
             fit = model.fit()
             forecast = fit.forecast(periods)
             
             history_points = [{'date': d.strftime('%Y-%m-%d'), 'value': float(v)} for d, v in ts.items()]
             forecast_points = [{'date': d.strftime('%Y-%m-%d'), 'value': float(v)} for d, v in forecast.items()]
             
             return {
                'success': True,
                'metric': 'turnover_rate',
                'history': history_points,
                'forecast': forecast_points
            }
        except Exception as e:
            return {'success': False, 'reason': str(e)}

