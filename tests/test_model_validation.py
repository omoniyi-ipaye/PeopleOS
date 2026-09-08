"""Model evaluation isolation and deliberately bad-model controls."""
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.model_training import RawFeatures, binary_metrics, train_attrition_model
from src.ml_engine import MLEngineError
from src.platform.model_lifecycle import ModelEvaluationPolicy


def valid_metrics():
    return dict(roc_auc=.8, brier_score=.12, baseline_brier_score=.25,
                average_precision=.8, baseline_average_precision=.5,
                calibration_error=.05, test_size=100, test_class_counts={'0': 50, '1': 50},
                cv_preprocessing_fold_local=True, holdout_untouched_by_fit=True)


@pytest.mark.parametrize('field,bad', [('baseline_brier_score', .1), ('calibration_error', .3),
                                     ('test_size', 20), ('test_class_counts', {'0': 99, '1': 1}),
                                     ('cv_preprocessing_fold_local', False), ('roc_auc', np.nan)])
def test_activation_rejects_unreliable_evidence(field, bad):
    metrics = valid_metrics()
    assert ModelEvaluationPolicy().evaluate(metrics)['passed']
    metrics[field] = bad
    assert not ModelEvaluationPolicy().evaluate(metrics)['passed']


def test_calibration_is_weighted_and_baseline_is_independent_of_test_prevalence():
    metrics = binary_metrics([0, 0, 0, 1], [.1, .1, .1, .6], .5)
    assert metrics['calibration_error'] == pytest.approx(.175)
    assert metrics['brier_score'] == pytest.approx(.0475)
    assert metrics['baseline_brier_score'] == .25


def test_each_cv_fold_fits_its_own_imputation_statistics():
    X = pd.DataFrame({'EmployeeID': [f'E{i}' for i in range(60)],
                      'Salary': np.arange(60, dtype=float) ** 2})
    X.loc[::5, 'Salary'] = np.nan
    y = np.tile([0, 1], 30)
    cv = list(StratifiedKFold(3, shuffle=True, random_state=12).split(X, y))
    pipeline = Pipeline([('features', RawFeatures()), ('model', DecisionTreeClassifier(random_state=1))])
    result = cross_validate(pipeline, X, y, cv=cv, return_estimator=True)
    for estimator, (train, _) in zip(result['estimator'], cv):
        actual = estimator.named_steps['features'].preprocessor_.impute_values['Salary']
        assert actual == X.iloc[train]['Salary'].median()


def test_prediction_missing_columns_preserves_fitted_schema():
    X = pd.DataFrame({'Salary': [100., 200., 300.], 'Tenure': [1., 2., 3.], 'Dept': ['A', 'B', 'A']})
    transform = RawFeatures().fit(X)
    result = transform.transform(pd.DataFrame({'Salary': [200.]}))
    assert result.columns.tolist() == transform.columns_
    assert np.isfinite(result.to_numpy()).all()


def test_target_proxies_and_employee_identifiers_are_not_features():
    X = pd.DataFrame({'EmployeeID': ['A', 'B'], 'EmployeeNumber': [1, 2],
                      'TerminationDate': ['2024-01-01', None], 'is_active': [0, 1],
                      'EmploymentStatus': ['Left', 'Active'], 'Salary': [100., 200.]})
    assert RawFeatures().fit(X).columns_ == ['Salary']


def test_unknown_outcomes_and_small_classes_are_rejected():
    X = pd.DataFrame({'EmployeeID': [f'E{i}' for i in range(40)], 'Salary': np.arange(40), 'Attrition': [0, 1] * 20})
    X.loc[0, 'Attrition'] = 2
    with pytest.raises(MLEngineError, match='unknown'):
        train_attrition_model(X)
    X['Attrition'] = [0] * 39 + [1]
    with pytest.raises(MLEngineError, match='ten'):
        train_attrition_model(X)


def test_model_lab_does_not_train_on_future_outcomes_to_claim_backtest():
    from src.model_lab_engine import ModelLabEngine
    lab = ModelLabEngine.__new__(ModelLabEngine)
    # No DB/model access is necessary to report the missing validation evidence.
    result = lab.backtest_flight_risk()
    assert result['status'] == 'warning'
    assert result['metrics'] is None
