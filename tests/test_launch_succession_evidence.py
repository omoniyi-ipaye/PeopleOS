"""Potential must come from recorded assessments, never performance proxies."""

import numpy as np
import pandas as pd

from src.succession_engine import SuccessionEngine


def workforce():
    return pd.DataFrame({
        'EmployeeID': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
        'Dept': ['Engineering'] * 8,
        'Tenure': [15, 15, .1, 4, 5, 6, 7, 8],
        'LastRating': [5., 5., 2., 5., 5., 5., 5., 5.],
        'Attrition': [0] * 8,
    })


def test_performance_and_tenure_cannot_create_potential_or_recommendations():
    engine = SuccessionEngine(workforce())
    result = engine.identify_high_potentials()
    assert result.empty
    assert result.attrs['assessment_status'] == 'unavailable'
    assert {'PotentialLevel', 'AttritionRisk', 'PotentialRating'} <= set(result.columns)
    assert engine.get_retention_recommendations() == []
    assert engine.get_9box_matrix().NineBox.eq('Unassessed').all()


def test_only_explicit_valid_high_potential_is_reported():
    frame = workforce()
    frame['PotentialRating'] = [None, 5., '4', 3.5, 0, 6, np.inf, 'invalid']
    engine = SuccessionEngine(frame)
    result = engine.identify_high_potentials().set_index('EmployeeID')
    assert set(result.index) == {'B', 'C'}
    assert result.loc['B', 'PotentialLevel'] == 'Star'
    assert result.loc['C', 'PotentialLevel'] == 'High'
    assert result.loc['C', 'Tenure'] == .1
    assert result.PotentialRating.to_dict() == {'B': 5., 'C': 4.}
    matrix = engine.get_9box_matrix().set_index('EmployeeID')
    assert matrix.loc['B', 'NineBox'] == 'Stars'
    assert matrix.loc['C', 'NineBox'] == 'Potential Gems'


def test_low_recorded_potential_is_not_overridden_by_performance():
    frame = workforce().assign(PotentialRating=2)
    result = SuccessionEngine(frame).identify_high_potentials()
    assert result.empty
    assert result.attrs['assessment_status'] == 'no_high_potential_assessments'


def test_departed_employee_is_not_a_current_potential_candidate():
    frame = workforce().assign(PotentialRating=5)
    frame.loc[0, 'Attrition'] = 1
    assert 'A' not in SuccessionEngine(frame).identify_high_potentials().EmployeeID.tolist()


def test_missing_performance_does_not_invent_star_label():
    frame = workforce().iloc[:1].assign(PotentialRating=5, LastRating=np.nan)
    result = SuccessionEngine(frame).identify_high_potentials()
    assert result.PotentialLevel.tolist() == ['High']
    assert result.LastRating.isna().all()
    frame['LastRating'] = pd.Series([pd.NA], dtype='Float64')
    assert SuccessionEngine(frame).identify_high_potentials().PotentialLevel.tolist() == ['High']
