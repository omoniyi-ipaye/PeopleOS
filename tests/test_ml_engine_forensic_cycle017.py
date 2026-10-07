"""Cycle 017 forensic contracts for MLEngine and predictive release boundaries."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import src.ml_engine as ml_module
from src.ml_engine import MLEngine, MLEngineError
from src.model_training import train_attrition_model


def matrix(n: int = 80) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(17017)
    X = pd.DataFrame({
        'Tenure': rng.uniform(0.5, 12, n),
        'Salary': rng.uniform(40000, 120000, n),
        'LastRating': rng.uniform(2, 5, n),
        'Age': rng.integers(22, 62, n),
    })
    y = pd.Series(([0, 1] * (n // 2)) + ([0] if n % 2 else []), index=X.index)
    return X, y


def raw_workforce(n: int = 320) -> pd.DataFrame:
    rng = np.random.default_rng(17170)
    outcome = np.tile([0, 1], n // 2)
    return pd.DataFrame({
        'EmployeeID': [f'E{i:04d}' for i in range(n)],
        'Dept': np.where(np.arange(n) % 2, 'Engineering', 'People'),
        'Tenure': rng.uniform(0.5, 12, n),
        'Salary': rng.uniform(40000, 120000, n),
        'LastRating': rng.uniform(2, 5, n),
        'Age': rng.integers(22, 62, n),
        'Gender': np.where(np.arange(n) % 2, 'Female', 'Male'),
        'JobTitle': np.where(np.arange(n) % 3, 'Specialist', 'Manager'),
        'Location': np.where(np.arange(n) % 2, 'Madrid', 'London'),
        'Attrition': outcome,
    })


def test_invalid_or_reversed_risk_threshold_config_fails_closed(monkeypatch):
    invalid = [
        {'risk_threshold_medium': .8, 'risk_threshold_high': .7},
        {'risk_threshold_medium': -.1, 'risk_threshold_high': .7},
        {'risk_threshold_medium': .5, 'risk_threshold_high': 1.1},
        {'risk_threshold_medium': np.nan, 'risk_threshold_high': .7},
        {'risk_threshold_medium': True, 'risk_threshold_high': .7},
        {'risk_threshold_medium': '0.5', 'risk_threshold_high': .7},
    ]
    for values in invalid:
        monkeypatch.setattr(ml_module, 'load_config', lambda values=values: {'ml': values})
        with pytest.raises(MLEngineError, match='risk thresholds'):
            MLEngine()


def test_low_level_training_rejects_single_class_and_clears_stale_state():
    X, _ = matrix(80)
    engine = MLEngine()
    engine.model = object()
    engine.is_trained = True
    engine.feature_names = ['stale']
    engine.shap_explainer = object()
    engine.shap_values = np.array([1.0])
    with pytest.raises(MLEngineError, match='two observed classes'):
        engine.train_model(X, pd.Series([0] * len(X)))
    assert engine.model is None
    assert engine.is_trained is False
    assert engine.feature_names == []
    assert engine.shap_explainer is None
    assert engine.shap_values is None


def test_low_level_training_rejects_under_supported_class_before_fit():
    X, _ = matrix(60)
    y = pd.Series([0] * 55 + [1] * 5)
    engine = MLEngine()
    with pytest.raises(MLEngineError, match='at least 10 observations in each class'):
        engine.train_model(X, y)
    assert engine.model is None and not engine.is_trained


def test_low_level_training_rejects_nonfinite_features_and_misaligned_target():
    X, y = matrix(60)
    bad = X.copy(); bad.loc[0, 'Salary'] = np.inf
    with pytest.raises(MLEngineError, match='finite numeric feature matrix'):
        MLEngine().train_model(bad, y)
    with pytest.raises(MLEngineError, match='aligned'):
        MLEngine().train_model(X, y.iloc[:-1])


def test_internal_probability_scoring_requires_exact_feature_contract():
    engine = MLEngine()
    engine.is_trained = True
    engine.feature_names = ['Salary', 'Tenure']
    engine.model = SimpleNamespace(
        classes_=np.array([0, 1]),
        predict_proba=lambda X: np.tile([[.7, .3]], (len(X), 1)),
    )
    valid = pd.DataFrame({'Salary': [50000.], 'Tenure': [2.]})
    assert engine.predict_risk(valid).tolist() == [.3]
    with pytest.raises(MLEngineError, match='exact trained feature contract'):
        engine.predict_risk(valid[['Tenure', 'Salary']])
    with pytest.raises(MLEngineError, match='exact trained feature contract'):
        engine.predict_risk(valid.assign(Age=30))


def test_internal_probability_scoring_rejects_nonfinite_features():
    engine = MLEngine()
    engine.is_trained = True
    engine.feature_names = ['Salary']
    engine.model = SimpleNamespace(classes_=np.array([0, 1]), predict_proba=lambda X: [[.5, .5]])
    with pytest.raises(MLEngineError, match='finite numeric features'):
        engine.predict_risk(pd.DataFrame({'Salary': [np.inf]}))


def test_legacy_individual_prediction_surface_is_disabled():
    engine = MLEngine()
    engine.is_trained = True
    engine.model = object()
    with pytest.raises(MLEngineError, match='Individual prediction output is disabled'):
        engine.predict(pd.DataFrame({'EmployeeID': ['E1'], 'Salary': [50000]}))


def test_employee_action_recommendations_are_disabled_at_engine_boundary():
    engine = MLEngine()
    drivers = [{'feature': 'Salary', 'contribution': .4, 'value': 40000, 'abs_contribution': .4}]
    assert engine.get_recommendations('E001', .95, drivers) == []


def test_feature_importance_fails_closed_on_misalignment_or_nonfinite_values():
    engine = MLEngine(); engine.is_trained = True; engine.feature_names = ['Salary', 'Tenure']
    engine.model = SimpleNamespace(feature_importances_=np.array([1.0]))
    with pytest.raises(MLEngineError, match='Feature importance'):
        engine.get_feature_importance_summary()
    engine.model = SimpleNamespace(feature_importances_=np.array([np.nan, 1.0]))
    with pytest.raises(MLEngineError, match='Feature importance'):
        engine.get_feature_importance_summary()


def test_rejected_retrospective_model_cannot_be_used_for_direct_scoring(monkeypatch):
    original_init = MLEngine.__init__
    def forest_only(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.models_to_compare = ['random_forest']
    monkeypatch.setattr(MLEngine, '__init__', forest_only)
    monkeypatch.setattr(MLEngine, '_prepare_shap', lambda *args: None)
    outcome = np.tile([0, 1], 160)
    frame = raw_workforce(320)
    frame['Salary'] = 50000.0
    frame['Tenure'] = 2.0
    frame['LastRating'] = 3.0
    frame['Age'] = 35
    frame['Dept'] = 'People'
    frame['Gender'] = 'Unknown'
    frame['JobTitle'] = 'Specialist'
    frame['Location'] = 'Madrid'
    frame['Attrition'] = outcome
    artifact = train_attrition_model(frame)
    assert artifact.metrics['reliability'] == 'Insufficient validation'
    assert artifact.metrics['future_departure_validated'] is False
    assert artifact.engine.is_trained is False
    with pytest.raises(MLEngineError, match='Model not trained'):
        artifact.engine.predict_risk(pd.DataFrame({column: [0.0] for column in artifact.engine.feature_names}))
