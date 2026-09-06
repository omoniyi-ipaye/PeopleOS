"""Leakage-safe predictive training boundary for PeopleOS.

Raw employee rows are split before any learned preprocessing. Imputation,
outlier bounds, categorical encodings and scaling are fitted only on training
rows and then reused unchanged on the holdout. Model-selection CV keeps SMOTE
inside folds through MLEngine's governed training helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.ml_engine import MLEngine, MLEngineError
from src.population import normalize_attrition, resolve_current_population
from src.preprocessor import Preprocessor


@dataclass
class TrainedModelArtifact:
    engine: MLEngine
    metrics: dict[str, Any]
    train_employee_ids: list[str]
    test_employee_ids: list[str]


def _prepare_raw(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    current, _ = resolve_current_population(df)
    if 'Attrition' not in current.columns:
        raise MLEngineError('Predictive training requires an Attrition outcome')
    target = normalize_attrition(current['Attrition'])
    valid = target.notna()
    current = current.loc[valid].copy()
    target = target.loc[valid].astype(int)
    if target.nunique() != 2:
        raise MLEngineError('Predictive training requires both active (0) and departed (1) Attrition outcomes')
    counts = target.value_counts()
    if counts.min() < 2:
        raise MLEngineError('Each Attrition class requires at least two observations for a holdout split')
    current['Attrition'] = target
    return current, target


def train_attrition_model(df: pd.DataFrame) -> TrainedModelArtifact:
    raw, target = _prepare_raw(df)
    engine = MLEngine()

    indices = np.arange(len(raw))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=engine.test_split_ratio,
        random_state=engine.random_seed,
        stratify=target,
    )
    raw_train = raw.iloc[train_idx].copy()
    raw_test = raw.iloc[test_idx].copy()
    y_train = target.iloc[train_idx].reset_index(drop=True)
    y_test = target.iloc[test_idx].reset_index(drop=True)

    preprocessor = Preprocessor()
    train_processed, metadata = preprocessor.fit_transform(raw_train, target_column='Attrition')
    test_processed = preprocessor.transform(raw_test, target_column='Attrition')
    feature_cols = [c for c in preprocessor.numeric_columns + preprocessor.categorical_columns if c in train_processed.columns and c != 'Attrition']
    if not feature_cols:
        raise MLEngineError('No usable predictive features remain after preprocessing')
    X_train = train_processed[feature_cols].select_dtypes(include=[np.number]).copy()
    X_test = test_processed.reindex(columns=X_train.columns).copy()
    if X_train.empty:
        raise MLEngineError('No numeric predictive features remain after preprocessing')

    sample_warnings = engine._validate_sample_size(X_train, y_train)
    candidate_models = ['random_forest']
    from src.ml_engine import XGB_AVAILABLE, LGBM_AVAILABLE
    if XGB_AVAILABLE:
        candidate_models.append('xgboost')
    if LGBM_AVAILABLE:
        candidate_models.append('lightgbm')

    scores: dict[str, float] = {}
    best_model_type = 'random_forest'
    best_f1 = -1.0
    for model_type in candidate_models:
        score = engine._evaluate_model_type_with_smote(model_type, X_train, y_train)
        scores[model_type] = score
        if score > best_f1:
            best_f1 = score
            best_model_type = model_type

    engine.model = engine._tune_and_train_with_smote(best_model_type, X_train, y_train)
    engine.best_model_name = best_model_type
    engine.feature_names = list(X_train.columns)
    engine.preprocessor = preprocessor
    engine.is_trained = True

    y_pred = engine.model.predict(X_test)
    y_proba = engine.model.predict_proba(X_test)[:, 1]
    warnings = list(sample_warnings)
    metrics: dict[str, Any] = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_test, y_proba)),
        'brier_score': float(brier_score_loss(y_test, y_proba)),
        'best_model': best_model_type,
        'candidate_cv_f1': scores,
        'train_size': int(len(X_train)),
        'test_size': int(len(X_test)),
        'preprocessing_fit_scope': 'training_rows_only',
        'holdout_untouched_by_fit': True,
        'feature_count': len(X_train.columns),
        'warnings': warnings,
    }
    try:
        true_prob, pred_prob = calibration_curve(y_test, y_proba, n_bins=min(5, max(2, len(y_test) // 10)))
        calibration_error = float(np.mean(np.abs(true_prob - pred_prob)))
        metrics['calibration_error'] = calibration_error
        if calibration_error > .15:
            warnings.append('Holdout calibration error exceeds 15%; model probabilities should not be interpreted as precise individual likelihoods.')
    except ValueError:
        metrics['calibration_error'] = None
        warnings.append('Holdout sample could not support a stable calibration estimate.')

    if hasattr(engine.model, 'feature_importances_'):
        metrics['feature_importances'] = dict(zip(engine.feature_names, engine.model.feature_importances_.tolist()))

    if len(sample_warnings) > 1 or any('calibration' in w.lower() for w in warnings):
        metrics['reliability'] = 'Low'
    elif sample_warnings:
        metrics['reliability'] = 'Medium'
    else:
        metrics['reliability'] = 'High'

    # SHAP background is training-only; holdout rows never become explanation background.
    engine._prepare_shap(X_train)
    metrics['warnings'] = list(dict.fromkeys(warnings))
    train_ids = raw_train['EmployeeID'].astype(str).tolist() if 'EmployeeID' in raw_train.columns else []
    test_ids = raw_test['EmployeeID'].astype(str).tolist() if 'EmployeeID' in raw_test.columns else []
    return TrainedModelArtifact(engine=engine, metrics=metrics, train_employee_ids=train_ids, test_employee_ids=test_ids)
