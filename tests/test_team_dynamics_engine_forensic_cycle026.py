"""Cycle 026 aggregate denominator and missing-department contracts."""

import numpy as np
import pandas as pd


def staff(n=20):
    return pd.DataFrame({
        'EmployeeID': [f'E{i}' for i in range(n)],
        'Dept': 'A',
        'Tenure': 2.0,
        'Salary': 100.0,
        'LastRating': 4.0,
        'Age': 30,
        'Attrition': 0,
    })


def test_unknown_departments_reconcile_to_active_composition():
    from src.team_dynamics_engine import TeamDynamicsEngine

    frame = staff()
    frame.loc[:9, 'Dept'] = None
    result = TeamDynamicsEngine(frame).get_team_composition()

    assert result['Headcount'].sum() == 20
    assert result.loc[result['Dept'] == 'Unknown', 'Headcount'].iloc[0] == 10


def test_unmeasured_team_health_is_unavailable_not_healthy():
    from src.team_dynamics_engine import TeamDynamicsEngine

    frame = staff().drop(columns=['Tenure', 'LastRating', 'Attrition'])
    result = TeamDynamicsEngine(frame).calculate_team_health_scores()

    assert result['HealthScore'].isna().all()
    assert result['Status'].eq('Unavailable').all()
