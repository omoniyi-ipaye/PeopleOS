"""Cycle 014 forensic contracts for CausalEngine.

Causal effects must remain unavailable unless PeopleOS has an explicit validated
identification design. These tests defend that boundary against accidental
correlation fallbacks, input reflection, privacy leakage and recommendation drift.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.causal_engine import CausalEngine


def workforce(n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(14014)
    return pd.DataFrame({
        'EmployeeID': [f'E{i:04d}' for i in range(n)],
        'Attrition': rng.integers(0, 2, n),
        'Salary': rng.normal(75000, 12000, n),
        'Age': rng.integers(21, 65, n),
        'ManagerID': [f'M{i // 8:03d}' for i in range(n)],
        'NationalID': [f'SECRET-{i:04d}' for i in range(n)],
    })


def assert_unavailable(result: dict) -> None:
    assert result['success'] is False
    assert result['available'] is False
    assert result['estimated_effect'] is None
    assert result['confidence_interval'] is None
    assert 'identification design' in result['reason'].lower()
    json.dumps(result, allow_nan=False)


def test_causal_effect_is_unavailable_for_plausible_hr_variables():
    engine = CausalEngine(workforce())
    assert_unavailable(engine.estimate_intervention_effect('Salary', 'Attrition', ['Age']))


@pytest.mark.parametrize('treatment,outcome,confounders', [
    ('Salary', 'Attrition', []),
    ('ManagerID', 'Attrition', ['Age']),
    ('NationalID', 'Attrition', ['EmployeeID']),
    ('__proto__', 'constructor', ['NationalID']),
    ('', '', None),
    ('Salary', 'Salary', ['Salary']),
])
def test_adversarial_or_invalid_requests_never_enable_a_causal_estimate(treatment, outcome, confounders):
    result = CausalEngine(workforce()).estimate_intervention_effect(treatment, outcome, confounders)
    assert_unavailable(result)


def test_causal_response_does_not_echo_sensitive_or_untrusted_field_names():
    engine = CausalEngine(workforce())
    result = engine.estimate_intervention_effect('NationalID', '<script>alert(1)</script>', ['EmployeeID'])
    payload = json.dumps(result)
    assert 'NationalID' not in payload
    assert 'EmployeeID' not in payload
    assert '<script>' not in payload
    assert 'SECRET-' not in payload


def test_recommendations_are_always_empty_without_identification_design():
    engine = CausalEngine(workforce())
    assert engine.get_intervention_recommendations() == []
    assert engine.get_intervention_recommendations() == []


def test_source_frame_is_not_mutated_by_causal_calls():
    frame = workforce()
    before = frame.copy(deep=True)
    engine = CausalEngine(frame)
    engine.estimate_intervention_effect('Salary', 'Attrition', ['Age'])
    engine.get_intervention_recommendations()
    pd.testing.assert_frame_equal(frame, before)


def test_engine_state_isolated_from_later_source_frame_mutation():
    frame = workforce()
    engine = CausalEngine(frame)
    original = engine.df.copy(deep=True)
    frame.loc[:, 'Salary'] = 0
    frame.loc[:, 'NationalID'] = 'CHANGED'
    pd.testing.assert_frame_equal(engine.df, original)


def test_result_is_dataset_invariant_because_no_effect_is_claimed():
    a = CausalEngine(workforce(60)).estimate_intervention_effect('Salary', 'Attrition', ['Age'])
    bframe = workforce(600)
    bframe['Salary'] = np.linspace(-1e300, 1e300, len(bframe))
    b = CausalEngine(bframe).estimate_intervention_effect('Salary', 'Attrition', ['Age'])
    assert a == b
    assert_unavailable(a)


def test_no_numeric_or_probability_like_causal_claim_can_leak_into_result():
    result = CausalEngine(workforce()).estimate_intervention_effect('Salary')
    forbidden = {'effect', 'effect_size', 'p_value', 'probability', 'confidence', 'risk_reduction', 'uplift'}
    assert forbidden.isdisjoint(result.keys())
    assert_unavailable(result)
