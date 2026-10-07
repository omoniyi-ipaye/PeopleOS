"""Cycle 018 forensic contracts for read-only ModelLab diagnostics."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes import model_lab as routes
from src.model_lab_engine import ModelLabEngine, ModelLabError


class FixedPreprocessor:
    def __init__(self, result: pd.DataFrame):
        self.result = result

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.result.copy(deep=True)


class FixedEngine:
    is_trained = True
    feature_names = ['Salary', 'Tenure']

    def __init__(self, transformed: pd.DataFrame | None = None):
        self.preprocessor = FixedPreprocessor(
            transformed if transformed is not None else pd.DataFrame({
                'Salary': [0.0, 1.0, 2.0, 3.0],
                'Tenure': [0.0, 1.0, 2.0, 3.0],
            })
        )

    def get_feature_importance_summary(self) -> pd.DataFrame:
        return pd.DataFrame({'feature': ['Salary', 'Tenure'], 'importance': [0.6, 0.4]})

    def get_risk_category(self, score: float) -> str:
        return 'High' if score >= 0.75 else 'Medium' if score >= 0.5 else 'Low'


def test_model_lab_has_no_implicit_database_or_model_and_backtest_stays_unavailable():
    lab = ModelLabEngine()
    assert lab.db is None and lab.ml_engine is None
    assert lab.analyze_feature_sensitivity() == []
    result = lab.backtest_flight_risk()
    assert result['metrics'] is None
    assert 'timestamped predictions' in result['message']


def test_diagnostics_are_read_only_and_never_recommend_automatic_pruning():
    source = pd.DataFrame({'EmployeeID': ['E1', 'E2', 'E3', 'E4']})
    original = source.copy(deep=True)
    lab = ModelLabEngine(ml_engine=FixedEngine(), data=source)
    diagnostics = lab.analyze_feature_sensitivity()
    pd.testing.assert_frame_equal(source, original)
    assert diagnostics[0]['status'] == 'High correlation observed'
    assert all('no feature change was applied' in item['recommendation'].lower() for item in diagnostics)
    plan = lab.generate_refinement_plan()
    assert plan['status'] == 'review_only'
    assert plan['automated_features_to_prune'] == []
    assert plan['metrics']['estimated_accuracy_lift'].startswith('Unknown')
    assert all('prune ' not in action.lower() and 'remove ' not in action.lower() for action in plan['suggested_actions'])


def test_single_observation_is_not_labelled_stable():
    transformed = pd.DataFrame({'Salary': [1.0], 'Tenure': [2.0]})
    result = ModelLabEngine(
        ml_engine=FixedEngine(transformed), data=pd.DataFrame({'EmployeeID': ['E1']})
    ).analyze_feature_sensitivity()
    assert {item['status'] for item in result} == {'Insufficient observations'}
    assert {item['reliability'] for item in result} == {0.0}


def test_diagnostics_fail_closed_on_nonfinite_or_mismatched_evidence():
    nonfinite = pd.DataFrame({'Salary': [1.0, np.inf], 'Tenure': [1.0, 2.0]})
    with pytest.raises(ModelLabError, match='finite transformed measurements'):
        ModelLabEngine(ml_engine=FixedEngine(nonfinite), data=pd.DataFrame({'EmployeeID': ['E1', 'E2']})).analyze_feature_sensitivity()
    missing = pd.DataFrame({'Salary': [1.0, 2.0]})
    with pytest.raises(ModelLabError, match='feature contract'):
        ModelLabEngine(ml_engine=FixedEngine(missing), data=pd.DataFrame({'EmployeeID': ['E1', 'E2']})).analyze_feature_sensitivity()


def test_model_lab_route_uses_active_runtime_model_and_blocks_provenance_mismatch(monkeypatch):
    frame = pd.DataFrame({'EmployeeID': ['E1', 'E2', 'E3', 'E4']})
    engine = FixedEngine()
    state = SimpleNamespace(
        raw_df=frame,
        ml_engine=engine,
        has_data=lambda: True,
        load_from_database=lambda: False,
    )
    monkeypatch.setattr(routes, 'validated_risk_scores', lambda current: pd.DataFrame({
        'EmployeeID': frame.EmployeeID, 'risk_score': [0.1, 0.2, 0.3, 0.4]
    }))
    app = FastAPI()
    app.dependency_overrides[routes.get_app_state] = lambda: state
    app.include_router(routes.router)
    with TestClient(app) as client:
        response = client.get('/api/model-lab/sensitivity')
        assert response.status_code == 200
        assert response.json()[0]['feature'] == 'Salary'
        optimized = client.post('/api/model-lab/optimize')
        assert optimized.status_code == 200
        assert optimized.json()['plan_applied'] is False

    def mismatch(current):
        raise routes.IntegrityError('Model output belongs to a different dataset snapshot.')

    monkeypatch.setattr(routes, 'validated_risk_scores', mismatch)
    with TestClient(app) as client:
        response = client.get('/api/model-lab/sensitivity')
        assert response.status_code == 409

