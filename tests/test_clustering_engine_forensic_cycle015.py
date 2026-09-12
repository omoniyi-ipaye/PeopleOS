"""Cycle 015 forensic contracts for ClusteringEngine."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.clustering_engine import ClusteringEngine


def workforce(n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(15015)
    frame = pd.DataFrame({
        'EmployeeID': [f'E{i:04d}' for i in range(n)],
        'Attrition': [0] * n,
        'Salary': np.r_[rng.normal(50000, 2000, n // 2), rng.normal(100000, 2000, n - n // 2)],
        'Tenure': np.r_[rng.normal(2, .2, n // 2), rng.normal(8, .2, n - n // 2)],
        'LastRating': rng.uniform(2.5, 5, n),
        'Age': rng.integers(24, 60, n),
        'Dept': ['People'] * (n // 4) + ['Engineering'] * (n - n // 4),
    })
    return frame


def test_training_output_is_aggregate_only_and_does_not_expose_employee_ids():
    result = ClusteringEngine(workforce()).train(n_clusters=2, auto_tune=False)
    assert result['success'] is True
    assert 'labels' not in result
    payload = json.dumps(result, allow_nan=False)
    assert 'E0000' not in payload
    assert 'EmployeeID' not in payload


def test_employee_cluster_membership_is_disabled_at_engine_boundary():
    engine = ClusteringEngine(workforce())
    assert engine.train(n_clusters=2, auto_tune=False)['success']
    assert engine.get_employee_clusters().empty


def test_small_clusters_are_suppressed_from_aggregate_outputs():
    frame = workforce(40)
    frame.loc[:2, 'Salary'] = 1_000_000
    frame.loc[:2, 'Tenure'] = 30
    result = ClusteringEngine(frame).train(n_clusters=3, auto_tune=False)
    assert result['success'] is True
    assert all(int(v) >= 5 for v in result['cluster_counts'].values())
    assert result['suppressed_cluster_count'] >= 1
    assert set(map(int, result['feature_summary'].keys())) == set(map(int, result['cluster_counts'].keys()))


def test_population_and_complete_case_coverage_are_disclosed():
    frame = workforce(60)
    frame.loc[:9, 'Salary'] = np.nan
    result = ClusteringEngine(frame).train(n_clusters=2, auto_tune=False)
    assert result['success'] is True
    pop = result['population']
    assert pop['source_population'] == 60
    assert pop['analysis_population'] == 50
    assert pop['excluded_missing_or_nonfinite_features'] == 10
    assert pop['coverage'] == pytest.approx(50 / 60)


def test_nonfinite_and_extreme_finite_values_never_escape_as_json_numbers():
    frame = workforce(80)
    frame['Salary'] = np.linspace(1e300, 1.1e300, len(frame))
    result = ClusteringEngine(frame).train(n_clusters=2, auto_tune=False)
    json.dumps(result, allow_nan=False)
    if result['success']:
        assert all(np.isfinite(float(v)) for values in result['feature_summary'].values() for v in values.values())


def test_source_frame_is_not_mutated():
    frame = workforce()
    before = frame.copy(deep=True)
    ClusteringEngine(frame).train(n_clusters=2, auto_tune=False)
    pd.testing.assert_frame_equal(frame, before)


def test_failed_retrain_clears_previous_model_and_aggregate_results():
    engine = ClusteringEngine(workforce())
    assert engine.train(n_clusters=2, auto_tune=False)['success']
    engine.df['Salary'] = 1
    engine.df['Tenure'] = 1
    engine.df['LastRating'] = 1
    engine.df['Age'] = 1
    failed = engine.train()
    assert failed['success'] is False
    assert engine.model is None
    assert engine.results == {}
    assert engine.get_employee_clusters().empty


def test_cluster_ids_are_not_given_stable_persona_or_risk_semantics():
    result = ClusteringEngine(workforce()).train(n_clusters=2, auto_tune=False)
    assert result['success']
    assert result['cluster_semantics'] == 'unsupervised_group_ids_are_arbitrary_and_not_stable_personas_or_risk_levels'
    assert 'cluster_descriptions' not in result


def test_invalid_cluster_counts_fail_closed():
    frame = workforce(30)
    for bad in [True, False, 0, 1, -1, 1.5, '3', 30, 31]:
        result = ClusteringEngine(frame).train(n_clusters=bad, auto_tune=False)
        assert result['success'] is False


def test_auto_tune_never_selects_cluster_count_that_creates_publishable_small_cells():
    frame = workforce(35)
    result = ClusteringEngine(frame).train(auto_tune=True)
    assert result['success']
    assert all(int(v) >= 5 for v in result['cluster_counts'].values())


def test_feature_set_cannot_include_identifiers_even_if_numeric():
    frame = workforce()
    frame['ManagerID'] = np.arange(len(frame))
    frame['NationalID'] = np.arange(len(frame)) + 10000
    engine = ClusteringEngine(frame)
    result = engine.train(n_clusters=2, auto_tune=False)
    assert result['success']
    assert 'ManagerID' not in engine.feature_cols
    assert 'NationalID' not in engine.feature_cols


def test_failure_reason_does_not_echo_sensitive_values():
    frame = workforce(12)
    frame['Salary'] = np.nan
    frame['NationalID'] = [f'SECRET-{i}' for i in range(len(frame))]
    result = ClusteringEngine(frame).train()
    assert result['success'] is False
    assert 'SECRET-' not in result['reason']


def test_cluster_counts_reconcile_to_publishable_analysis_population():
    result = ClusteringEngine(workforce()).train(n_clusters=2, auto_tune=False)
    assert result['success']
    assert sum(result['cluster_counts'].values()) + result['suppressed_row_count'] == result['population']['analysis_population']
