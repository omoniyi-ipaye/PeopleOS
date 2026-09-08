"""Reproducible retrospective classification with fold-local preprocessing.

A random employee holdout measures classification of observed outcomes. It
cannot establish future departure accuracy without timestamped pre-outcome
features, a defined prediction horizon, and independent temporal validation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from src.ml_engine import MLEngine, MLEngineError
from src.population import normalize_attrition, resolve_current_population
from src.preprocessor import Preprocessor


@dataclass
class TrainedModelArtifact:
    engine: MLEngine
    metrics: dict[str, Any]
    train_employee_ids: list[str]
    test_employee_ids: list[str]


class RawFeatures(BaseEstimator, TransformerMixin):
    """Cloneable transformer: each CV fold learns its own preprocessing state."""
    def fit(self, X, y=None):
        self.preprocessor_ = Preprocessor()
        processed, _ = self.preprocessor_.fit_transform(X, target_column='Attrition')
        self.columns_ = [c for c in self.preprocessor_.numeric_columns + self.preprocessor_.categorical_columns
                         if c in processed and c != 'Attrition']
        if not self.columns_:
            raise MLEngineError('No usable predictive features remain')
        return self

    def transform(self, X):
        return self.preprocessor_.transform(X, target_column='Attrition')[self.columns_]


def _prepare_raw(df):
    current, _ = resolve_current_population(df)
    if 'EmployeeID' not in current or current['EmployeeID'].isna().any():
        raise MLEngineError('Predictive training requires non-missing employee identifiers')
    if 'Attrition' not in current:
        raise MLEngineError('Predictive training requires an Attrition outcome')
    target = normalize_attrition(current['Attrition'])
    if target.isna().any():
        raise MLEngineError('Resolve unknown Attrition outcomes before predictive training')
    target = target.astype(int)
    if target.nunique() != 2 or target.value_counts().min() < 10:
        raise MLEngineError('At least ten observations in each Attrition class are required')
    return current, target


def binary_metrics(y_true, probabilities, training_prevalence):
    """Holdout scores versus a constant training-prevalence baseline."""
    y = np.asarray(y_true)
    p = np.asarray(probabilities, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or not len(y) or len(y) != len(p):
        raise MLEngineError('Outcomes and probabilities must be non-empty aligned vectors')
    if not np.isin(y, [0, 1]).all():
        raise MLEngineError('Evaluation requires observed binary outcomes without missing values')
    if not np.isfinite(training_prevalence) or not 0 <= training_prevalence <= 1:
        raise MLEngineError('Training prevalence must be finite and between zero and one')
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise MLEngineError('Predicted probabilities must be finite, aligned, and between zero and one')
    y = y.astype(int)
    predicted = (p >= .5).astype(int)
    tp = int(((y == 1) & (predicted == 1)).sum())
    fn = int(((y == 1) & (predicted == 0)).sum())
    tn = int(((y == 0) & (predicted == 0)).sum())
    fp = int(((y == 0) & (predicted == 1)).sum())
    brier = float(brier_score_loss(y, p))
    baseline = float(np.mean((y - training_prevalence) ** 2))
    # Equal-width ECE, weighted by the number of observations in each bin.
    bins = np.minimum((p * 10).astype(int), 9)
    calibration = []
    for index in range(10):
        mask = bins == index
        if mask.any():
            calibration.append({'count': int(mask.sum()), 'mean_probability': float(p[mask].mean()),
                                'observed_share': float(y[mask].mean())})
    ece = sum(b['count'] * abs(b['mean_probability'] - b['observed_share']) for b in calibration) / len(y)
    return {
        'accuracy': float(accuracy_score(y, predicted)),
        'precision': float(precision_score(y, predicted, zero_division=0)),
        'recall': float(recall_score(y, predicted, zero_division=0)),
        'f1': float(f1_score(y, predicted, zero_division=0)),
        'roc_auc': float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        'average_precision': float(average_precision_score(y, p)),
        'baseline_average_precision': float(y.mean()),
        'brier_score': brier, 'baseline_brier_score': baseline,
        'brier_skill_score': 1 - brier / baseline if baseline > 0 else None,
        'calibration_error': float(ece), 'calibration_bins': calibration,
        'calibration_method': '10_equal_width_bins_weighted_by_sample_count',
        'test_class_counts': {str(c): int((y == c).sum()) for c in (0, 1)},
        'classification_threshold': .5,
        'threshold_selection': 'fixed_default_not_tuned_on_holdout',
        'confusion_matrix': {'true_negative': tn, 'false_positive': fp, 'false_negative': fn, 'true_positive': tp},
        'predicted_positive_count': tp + fp,
        'observed_positive_count': tp + fn,
        'recall_denominator': tp + fn,
        'majority_class_baseline_accuracy': float(np.mean(y == int(training_prevalence >= .5))),
        'majority_class_baseline_selection': 'majority_class_from_training_partition',
    }


def train_attrition_model(df: pd.DataFrame) -> TrainedModelArtifact:
    raw, target = _prepare_raw(df)
    engine = MLEngine()
    train_idx, test_idx = train_test_split(np.arange(len(raw)), test_size=engine.test_split_ratio,
                                         stratify=target, random_state=engine.random_seed)
    raw_train, raw_test = raw.iloc[train_idx].copy(), raw.iloc[test_idx].copy()
    y_train, y_test = target.iloc[train_idx], target.iloc[test_idx]
    # Target never enters transformer input, including during model selection.
    X_train, X_test = raw_train.drop(columns='Attrition'), raw_test.drop(columns='Attrition')
    folds = StratifiedKFold(n_splits=min(3, int(y_train.value_counts().min())),
                           shuffle=True, random_state=engine.random_seed)
    candidates = {}
    if 'random_forest' in engine.models_to_compare:
        candidates['random_forest'] = (RandomForestClassifier(random_state=engine.random_seed, n_jobs=1,
                                       class_weight='balanced' if engine.handle_imbalance else None),
                                      {'model__n_estimators': [100], 'model__max_depth': [5, None],
                                       'model__min_samples_leaf': [2, 5]})
    from src.ml_engine import XGB_AVAILABLE, LGBM_AVAILABLE
    if XGB_AVAILABLE and 'xgboost' in engine.models_to_compare:
        from xgboost import XGBClassifier
        candidates['xgboost'] = (XGBClassifier(random_state=engine.random_seed, n_jobs=1, eval_metric='logloss'),
                                {'model__n_estimators': [100], 'model__max_depth': [3, 5], 'model__learning_rate': [.05, .1]})
    if LGBM_AVAILABLE and 'lightgbm' in engine.models_to_compare:
        from lightgbm import LGBMClassifier
        candidates['lightgbm'] = (LGBMClassifier(random_state=engine.random_seed, n_jobs=1, verbose=-1),
                                 {'model__n_estimators': [100], 'model__num_leaves': [15, 31], 'model__learning_rate': [.05, .1]})
    if not candidates:
        raise MLEngineError('No configured candidate model is installed')
    searches = {}
    for name, (model, grid) in candidates.items():
        search = GridSearchCV(Pipeline([('features', RawFeatures()), ('model', model)]), grid,
                              scoring='average_precision', cv=folds, n_jobs=1, error_score='raise')
        search.fit(X_train, y_train)
        searches[name] = search
    best_name = max(searches, key=lambda name: searches[name].best_score_)
    best = searches[best_name].best_estimator_
    metrics = binary_metrics(y_test, best.predict_proba(X_test)[:, 1], float(y_train.mean()))
    metrics.update({
        'best_model': best_name, 'candidate_cv_average_precision': {n: float(s.best_score_) for n, s in searches.items()},
        'selected_parameters': searches[best_name].best_params_,
        'train_size': len(X_train), 'test_size': len(X_test),
        'preprocessing_fit_scope': 'each_cv_training_fold_then_full_training_partition',
        'cv_preprocessing_fold_local': True, 'holdout_untouched_by_fit': True,
        'evaluation_semantics': 'retrospective_employee_holdout_not_future_departure_validation',
        'future_departure_validated': False,
        'sampling_method': 'original_rows_no_synthetic_interpolation_of_category_codes',
        'random_seed': engine.random_seed,
    })
    warnings = engine._validate_sample_size(X_train, y_train)
    warnings.append('Retrospective classification only: future departure accuracy has not been validated.')
    if metrics['recall_denominator'] and metrics['recall'] < .5:
        counts = metrics['confusion_matrix']
        warnings.append(
            f"At the fixed 0.5 classification threshold the model detected {counts['true_positive']} "
            f"of {metrics['recall_denominator']} observed departures and missed {counts['false_negative']}. "
            'Overall accuracy does not establish adequate departure detection. '
            'Any alternative threshold must be selected using training validation data and evaluated independently.'
        )
    from src.platform.model_lifecycle import ModelEvaluationPolicy
    evaluation = ModelEvaluationPolicy().evaluate(metrics)
    metrics['reliability'] = 'Retrospective only' if evaluation['passed'] else 'Insufficient validation'
    metrics['validation_checks'] = evaluation['checks']
    if not evaluation['passed']:
        warnings.append('Model did not meet the minimum retrospective evaluation gate; activation is blocked.')
    metrics['warnings'] = warnings
    engine.model = best.named_steps['model']
    engine.preprocessor = best.named_steps['features'].preprocessor_
    engine.feature_names = best.named_steps['features'].columns_
    engine.best_model_name = best_name
    engine.is_trained = True
    metrics['feature_count'] = len(engine.feature_names)
    if hasattr(engine.model, 'feature_importances_'):
        metrics['feature_importances'] = dict(zip(engine.feature_names, engine.model.feature_importances_.tolist()))
    engine._prepare_shap(best.named_steps['features'].transform(X_train))
    return TrainedModelArtifact(engine, metrics, raw_train['EmployeeID'].astype(str).tolist(),
                                raw_test['EmployeeID'].astype(str).tolist())
