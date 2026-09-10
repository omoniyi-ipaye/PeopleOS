"""Cycle 012 forensic contracts for ForecastingEngine."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import pytest

from src.forecasting_engine import ForecastingEngine


def monthly_history(months=12, n=20, *, start='2024-01', salary=100000.0):
    frames=[]
    for period in pd.period_range(start, periods=months, freq='M'):
        date=period.to_timestamp(how='end').date().isoformat()
        frames.append(pd.DataFrame({
            'EmployeeID':[f'E{i}' for i in range(n)],
            'SnapshotDate':date,
            'Attrition':[0]*n,
            'Salary':[salary]*n,
            'PayPeriod':['annual']*n,
            'Currency':['EUR']*n,
        }))
    return pd.concat(frames, ignore_index=True)


def test_known_answer_constant_headcount_and_future_months():
    result=ForecastingEngine(monthly_history()).forecast_metric('headcount', periods=3)
    assert result['success'] is True
    assert [r['value'] for r in result['history']] == [20.0]*12
    assert [r['value'] for r in result['forecast']] == [20.0]*3
    assert [r['date'] for r in result['forecast']] == ['2025-01-31','2025-02-28','2025-03-31']
    assert all('lower' not in r and 'upper' not in r for r in result['forecast'])


@pytest.mark.parametrize('periods',[0,37,-1,1.5,True,False])
def test_invalid_horizons_fail_closed(periods):
    assert ForecastingEngine(monthly_history()).forecast_metric('headcount', periods=periods)['success'] is False


def test_missing_month_fails_closed():
    frame=monthly_history()
    frame=frame[~frame.SnapshotDate.str.startswith('2024-06')]
    assert ForecastingEngine(frame).forecast_metric('headcount')['success'] is False


def test_any_invalid_snapshot_date_fails_closed_instead_of_silently_shrinking_history():
    frame=monthly_history()
    frame.loc[0,'SnapshotDate']='not-a-date'
    result=ForecastingEngine(frame).forecast_metric('headcount')
    assert result['success'] is False
    assert 'date' in result['reason'].lower() or 'census' in result['reason'].lower()


def test_same_month_uses_latest_observed_census_independent_of_row_order():
    frame=monthly_history()
    extra=pd.DataFrame({
        'EmployeeID':[f'X{i}' for i in range(25)],
        'SnapshotDate':['2024-06-30']*25,
        'Attrition':[0]*25,
        'Salary':[100000.0]*25,
        'PayPeriod':['annual']*25,
        'Currency':['EUR']*25,
    })
    frame=pd.concat([frame,extra], ignore_index=True)
    a=ForecastingEngine(frame).forecast_metric('headcount')
    b=ForecastingEngine(frame.sample(frac=1,random_state=7)).forecast_metric('headcount')
    assert a['success'] and b['success']
    assert a['history'] == b['history']
    june=[r for r in a['history'] if r['date']=='2024-06-30'][0]
    assert june['value']==25.0


def test_conflicting_duplicate_employee_rows_in_same_snapshot_fail_closed_not_raise():
    frame=monthly_history()
    row=frame.iloc[[0]].copy()
    row['Salary']=99999.0
    frame=pd.concat([frame,row], ignore_index=True)
    result=ForecastingEngine(frame).forecast_metric('headcount')
    assert result['success'] is False
    assert 'conflict' in result['reason'].lower() or 'ambiguous' in result['reason'].lower()


def test_source_calendar_month_is_not_shifted_by_timezone_conversion():
    frame=monthly_history()
    frame['SnapshotDate']=pd.period_range('2024-01',periods=12,freq='M').repeat(20).astype(str) + '-01T00:30:00+01:00'
    result=ForecastingEngine(frame).forecast_metric('headcount', periods=1)
    assert result['success'] is True
    assert result['history'][0]['date']=='2024-01-31'
    assert result['history'][-1]['date']=='2024-12-31'


def test_salary_forecast_requires_complete_measured_active_salary_population_each_month():
    frame=monthly_history()
    mask=(frame.SnapshotDate.str.startswith('2024-06')) & (frame.EmployeeID=='E0')
    frame.loc[mask,'Salary']=np.nan
    result=ForecastingEngine(frame).forecast_metric('salary', periods=1)
    assert result['success'] is False
    assert 'salary' in result['reason'].lower() or 'coverage' in result['reason'].lower()


def test_salary_known_answer_preserves_units_and_mean():
    frame=monthly_history(salary=120000.0)
    result=ForecastingEngine(frame).forecast_metric('salary', periods=2)
    assert result['success'] is True
    assert all(r['value']==120000.0 for r in result['history'])
    assert all(r['value']==120000.0 for r in result['forecast'])


def test_salary_forecast_rejects_mixed_or_unconfirmed_units():
    frame=monthly_history()
    frame.loc[0,'Currency']='USD'
    assert ForecastingEngine(frame).forecast_metric('salary')['success'] is False
    frame=monthly_history().drop(columns=['PayPeriod'])
    assert ForecastingEngine(frame).forecast_metric('salary')['success'] is False


def test_unsupported_metric_and_frequency_fail_closed():
    engine=ForecastingEngine(monthly_history())
    assert engine.forecast_metric('turnover_rate')['success'] is False
    assert engine.forecast_metric('headcount',freq='D')['success'] is False


def test_output_is_json_finite_and_input_is_not_mutated():
    frame=monthly_history()
    before=frame.copy(deep=True)
    result=ForecastingEngine(frame).forecast_metric('headcount')
    json.dumps(result,allow_nan=False)
    pd.testing.assert_frame_equal(frame,before)


def test_row_order_does_not_change_forecast():
    frame=monthly_history(months=18)
    a=ForecastingEngine(frame).forecast_metric('headcount',periods=6)
    b=ForecastingEngine(frame.sample(frac=1,random_state=99)).forecast_metric('headcount',periods=6)
    assert a==b
