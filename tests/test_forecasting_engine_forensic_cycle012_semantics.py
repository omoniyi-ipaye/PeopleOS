from __future__ import annotations
import pandas as pd
from src.forecasting_engine import ForecastingEngine


def history():
    frames=[]
    for p in pd.period_range('2024-01',periods=12,freq='M'):
        frames.append(pd.DataFrame({
            'EmployeeID':[f'E{i}' for i in range(20)],
            'SnapshotDate':p.to_timestamp(how='end').date().isoformat(),
            'Attrition':[0]*20,
        }))
    return pd.concat(frames,ignore_index=True)


def test_headcount_forecast_requires_employee_identity_for_census_reconciliation():
    result=ForecastingEngine(history().drop(columns=['EmployeeID'])).forecast_metric('headcount')
    assert result['success'] is False
    assert 'employeeid' in result['reason'].lower() or 'identity' in result['reason'].lower()


def test_headcount_forecast_rejects_unknown_status_rows_in_any_selected_census():
    frame=history()
    frame.loc[(frame.SnapshotDate=='2024-06-30') & (frame.EmployeeID=='E0'),'Attrition']=pd.NA
    result=ForecastingEngine(frame).forecast_metric('headcount')
    assert result['success'] is False
    assert 'status' in result['reason'].lower() or 'attrition' in result['reason'].lower() or 'coverage' in result['reason'].lower()


def test_active_only_history_without_attrition_is_allowed_when_identity_is_complete():
    result=ForecastingEngine(history().drop(columns=['Attrition'])).forecast_metric('headcount',periods=1)
    assert result['success'] is True
    assert all(row['value']==20 for row in result['history'])
