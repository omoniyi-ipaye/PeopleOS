"""Adversarial HR-export columns must not manufacture ML validation evidence."""
import numpy as np
import pandas as pd
import pytest

from src.ml_engine import MLEngine, MLEngineError
from src.model_training import RawFeatures, train_attrition_model
from src.platform.model_lifecycle import ModelEvaluationPolicy


def test_unapproved_columns_are_excluded_and_audited():
    frame = pd.DataFrame({
        'Salary': [50000., 60000.], 'Dept': ['Sales', 'Finance'],
        'EmployeeID': ['E1', 'E2'], 'ExitInterviewCompleted': [0, 1],
        'custom_score': [0, 1], 'is_active': [1, 0],
        'RatingVelocity': [0, 1], 'SalaryGrowth': [0, 1],
    })
    features = RawFeatures().fit(frame)
    assert set(features.columns_) == {'Salary', 'Dept'}
    reasons = features.predictor_contract_['excluded_columns']
    assert reasons['ExitInterviewCompleted'] == 'outcome_or_post_outcome_field'
    assert reasons['custom_score'] == 'not_in_supported_predictor_contract'
    assert reasons['EmployeeID'] == 'identifier_not_predictor'
    assert reasons['is_active'] == 'outcome_or_post_outcome_field'
    assert reasons['RatingVelocity'] == 'not_in_supported_predictor_contract'


def test_inference_cannot_inject_unapproved_or_engineered_features():
    frame = pd.DataFrame({'Salary': [50000., 60000.], 'Tenure': [2., 3.]})
    features = RawFeatures().fit(frame)
    injected = frame.assign(StartingSalary=[1, 1], SalaryGrowth=[999, 999],
                            ExitInterviewCompleted=[0, 1], custom_score=[0, 1])
    pd.testing.assert_frame_equal(features.transform(frame), features.transform(injected))
    # The deployed engine uses its preprocessor directly, outside RawFeatures.
    assert features.preprocessor_.transform(injected).columns.tolist() == frame.columns.tolist()


def test_no_supported_predictors_fails_explicitly():
    with pytest.raises(MLEngineError, match='No usable predictive features'):
        RawFeatures().fit(pd.DataFrame({'ExitInterviewCompleted': [0, 1]}))


def test_duplicate_predictor_names_are_rejected():
    with pytest.raises(MLEngineError, match='unique column names'):
        RawFeatures().fit(pd.DataFrame([[1, 2]], columns=['Salary', 'Salary']))


def test_actual_training_cannot_pass_using_post_exit_outcome_copies(monkeypatch):
    # Independent expected answer: with constant legitimate predictors there is
    # no discrimination, even though either extra field reveals every outcome.
    original_init = MLEngine.__init__

    def forest_only(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.models_to_compare = ['random_forest']

    monkeypatch.setattr(MLEngine, '__init__', forest_only)
    monkeypatch.setattr(MLEngine, '_prepare_shap', lambda *args: None)
    outcome = np.tile([0, 1], 200)
    frame = pd.DataFrame({'EmployeeID': [f'E{i}' for i in range(400)],
                          'Salary': 50000., 'Dept': 'Sales', 'Attrition': outcome,
                          'ExitInterviewCompleted': outcome, 'custom_score': outcome})
    artifact = train_attrition_model(frame)
    assert artifact.metrics['roc_auc'] == .5
    assert not ModelEvaluationPolicy().evaluate(artifact.metrics)['passed']
    assert set(artifact.engine.feature_names) == {'Salary', 'Dept'}
    assert artifact.metrics['future_departure_validated'] is False
    assert artifact.metrics['predictor_contract']['pre_outcome_timing_verified'] is False
    assert 'ExitInterviewCompleted' in artifact.metrics['predictor_contract']['excluded_columns']
