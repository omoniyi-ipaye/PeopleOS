"""Governed predictive-model primitives for PeopleOS.

The engine supports retrospective attrition classification and internal probability
scoring for aggregate product surfaces. It does not expose employee-level prediction
lists or generate employment-action recommendations. Prospective future-departure
claims require independent temporal validation outside this module.
"""
from __future__ import annotations

from numbers import Real
from typing import Any, Optional

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, brier_score_loss, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except (ImportError, OSError):
    LGBM_AVAILABLE = False

import optuna
from optuna.samplers import TPESampler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from src.logger import get_logger
from src.preprocessor import Preprocessor
from src.utils import load_config

MIN_SAMPLES_FOR_ML = 50
MIN_SAMPLES_PER_CLASS = 10
logger = get_logger('ml_engine')


class MLEngineError(Exception):
    """Predictive-model contract or execution failure."""


class MLEngine:
    """Retrospective classification engine with fail-closed scoring contracts."""

    def __init__(self):
        self.config = load_config()
        self.ml_config = self.config.get('ml', {})
        self.random_seed = self.ml_config.get('random_seed', 42)
        self.shap_sample_size = self.ml_config.get('shap_sample_size', 100)
        self.risk_threshold_high = self.ml_config.get('risk_threshold_high', 0.75)
        self.risk_threshold_medium = self.ml_config.get('risk_threshold_medium', 0.50)
        self.test_split_ratio = self.ml_config.get('test_split_ratio', 0.2)
        self._validate_runtime_config()

        self.model: Optional[Any] = None
        self.best_model_name: str = ''
        self.feature_names: list[str] = []
        self.shap_explainer: Any = None
        self.shap_values: Optional[np.ndarray] = None
        self.is_trained = False
        self.handle_imbalance = bool(self.ml_config.get('handle_imbalance', True))
        configured_models = self.ml_config.get('models', ['random_forest', 'xgboost', 'lightgbm'])
        self.models_to_compare = list(configured_models) if isinstance(configured_models, (list, tuple)) else ['random_forest']
        self.n_trials = self.ml_config.get('optuna_trials', 10)
        self.preprocessor = Preprocessor()

    @staticmethod
    def _finite_real(value: Any) -> bool:
        return isinstance(value, Real) and not isinstance(value, (bool, np.bool_)) and np.isfinite(float(value))

    def _validate_runtime_config(self) -> None:
        medium, high = self.risk_threshold_medium, self.risk_threshold_high
        if not self._finite_real(medium) or not self._finite_real(high):
            raise MLEngineError('Configured risk thresholds must be finite numeric values')
        medium, high = float(medium), float(high)
        if not 0 <= medium < high <= 1:
            raise MLEngineError('Configured risk thresholds must satisfy 0 <= medium < high <= 1')
        self.risk_threshold_medium, self.risk_threshold_high = medium, high
        if not self._finite_real(self.test_split_ratio) or not 0 < float(self.test_split_ratio) < 1:
            raise MLEngineError('test_split_ratio must be a finite number strictly between zero and one')
        self.test_split_ratio = float(self.test_split_ratio)
        if isinstance(self.random_seed, (bool, np.bool_)) or not isinstance(self.random_seed, (int, np.integer)):
            raise MLEngineError('random_seed must be an integer')
        self.random_seed = int(self.random_seed)
        if isinstance(self.shap_sample_size, (bool, np.bool_)) or not isinstance(self.shap_sample_size, (int, np.integer)) or int(self.shap_sample_size) < 1:
            raise MLEngineError('shap_sample_size must be a positive integer')
        self.shap_sample_size = int(self.shap_sample_size)

    def _reset_model_state(self) -> None:
        self.model = None
        self.best_model_name = ''
        self.feature_names = []
        self.shap_explainer = None
        self.shap_values = None
        self.is_trained = False

    def train(self, df: pd.DataFrame) -> dict:
        """Train through the leakage-safe raw-data pipeline."""
        from src.model_training import train_attrition_model
        self._reset_model_state()
        try:
            artifact = train_attrition_model(df)
            self.__dict__.update(artifact.engine.__dict__)
            return artifact.metrics
        except Exception:
            self._reset_model_state()
            raise

    def predict(self, df: pd.DataFrame) -> list[dict]:
        """Legacy employee-level prediction output is intentionally disabled."""
        raise MLEngineError(
            'Individual prediction output is disabled. Use governed aggregate predictive surfaces.'
        )

    @staticmethod
    def _validate_feature_matrix(X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame) or X.empty:
            raise MLEngineError('Training requires a non-empty feature DataFrame')
        if not X.columns.is_unique:
            raise MLEngineError('Training requires unique feature column names')
        non_numeric = [column for column in X if not pd.api.types.is_numeric_dtype(X[column])]
        if non_numeric:
            raise MLEngineError('Legacy feature-matrix training requires a finite numeric feature matrix')
        values = X.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise MLEngineError('Legacy feature-matrix training requires a finite numeric feature matrix')

    @staticmethod
    def _validated_binary_target(X: pd.DataFrame, y: pd.Series) -> pd.Series:
        if not isinstance(y, pd.Series) or len(X) != len(y) or not X.index.equals(y.index):
            raise MLEngineError('Feature matrix and target must be aligned')
        numeric = pd.to_numeric(y, errors='coerce')
        if numeric.isna().any() or not np.isin(numeric, [0, 1]).all():
            raise MLEngineError('Attrition target requires two observed binary classes encoded as 0 and 1')
        target = numeric.astype(int)
        counts = target.value_counts()
        if set(counts.index) != {0, 1}:
            raise MLEngineError('Attrition target requires two observed classes')
        if len(X) < MIN_SAMPLES_FOR_ML:
            raise MLEngineError(f'Legacy training requires at least {MIN_SAMPLES_FOR_ML} observations')
        if counts.min() < MIN_SAMPLES_PER_CLASS:
            raise MLEngineError(f'Legacy training requires at least {MIN_SAMPLES_PER_CLASS} observations in each class')
        return target

    def train_model(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Legacy preprocessed-matrix training; retained only with explicit limits."""
        self._reset_model_state()
        try:
            self._validate_feature_matrix(X)
            y = self._validated_binary_target(X, y)
            self.feature_names = list(X.columns)
            sample_size_warnings = self._validate_sample_size(X, y)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_split_ratio, random_state=self.random_seed, stratify=y
            )
            X_train_original, y_train_original = X_train.copy(), y_train.copy()

            candidate_models = [name for name in self.models_to_compare if name == 'random_forest']
            if XGB_AVAILABLE and 'xgboost' in self.models_to_compare:
                candidate_models.append('xgboost')
            if LGBM_AVAILABLE and 'lightgbm' in self.models_to_compare:
                candidate_models.append('lightgbm')
            if not candidate_models:
                raise MLEngineError('No configured candidate model is installed')

            results: dict[str, float] = {}
            best_model_type: Optional[str] = None
            best_f1 = -np.inf
            for model_type in candidate_models:
                score = self._evaluate_model_type_with_smote(model_type, X_train_original, y_train_original)
                if not np.isfinite(score):
                    continue
                results[model_type] = float(score)
                if score > best_f1:
                    best_f1, best_model_type = float(score), model_type
            if best_model_type is None:
                raise MLEngineError('No candidate model produced a finite cross-validation score')

            self.model = self._tune_and_train_with_smote(best_model_type, X_train_original, y_train_original)
            self.best_model_name = best_model_type
            classes = np.asarray(getattr(self.model, 'classes_', []))
            if classes.ndim != 1 or set(classes.tolist()) != {0, 1}:
                raise MLEngineError('Trained classifier did not retain both binary outcome classes')

            y_pred = np.asarray(self.model.predict(X_test))
            probabilities = np.asarray(self.model.predict_proba(X_test), dtype=float)
            positive_index = int(np.flatnonzero(classes == 1)[0])
            y_proba = probabilities[:, positive_index]
            if not np.isfinite(y_proba).all() or ((y_proba < 0) | (y_proba > 1)).any():
                raise MLEngineError('Trained classifier emitted invalid probabilities')

            metrics = {
                'accuracy': float(accuracy_score(y_test, y_pred)),
                'precision': float(precision_score(y_test, y_pred, zero_division=0)),
                'recall': float(recall_score(y_test, y_pred, zero_division=0)),
                'f1': float(f1_score(y_test, y_pred, zero_division=0)),
                'best_model': best_model_type,
                'train_size': int(len(X_train)),
                'test_size': int(len(X_test)),
                'candidate_cv_f1': results,
            }
            metrics['roc_auc'] = float(roc_auc_score(y_test, y_proba)) if y_test.nunique() == 2 else None
            metrics['brier_score'] = float(brier_score_loss(y_test, y_proba))
            prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=min(5, len(y_test)))
            calibration_error = float(np.mean(np.abs(prob_true - prob_pred))) if len(prob_true) else None
            metrics['calibration_error'] = calibration_error

            if hasattr(self.model, 'feature_importances_'):
                importances = np.asarray(self.model.feature_importances_, dtype=float)
                if importances.shape != (len(self.feature_names),) or not np.isfinite(importances).all():
                    raise MLEngineError('Feature importance output does not align with the trained feature contract')
                metrics['feature_importances'] = dict(zip(self.feature_names, importances.tolist()))

            metrics.update({
                'reliability': 'Unvalidated legacy feature-matrix evaluation',
                'future_departure_validated': False,
                'cv_preprocessing_fold_local': False,
                'warnings': sample_size_warnings + [
                    'Use train(raw_df) for fold-local preprocessing and baseline evaluation. '
                    'This low-level feature-matrix method cannot verify upstream leakage.'
                ],
            })
            self._prepare_shap(X_train_original)
            self.is_trained = True
            return metrics
        except MLEngineError:
            self._reset_model_state()
            raise
        except Exception as exc:
            self._reset_model_state()
            logger.error('Model training failed: %s: %s', type(exc).__name__, exc)
            raise MLEngineError(f'Model training failed: {type(exc).__name__}') from exc

    def _validate_sample_size(self, X: pd.DataFrame, y: pd.Series) -> list[str]:
        warnings: list[str] = []
        total_samples = len(X)
        if total_samples < MIN_SAMPLES_FOR_ML:
            warnings.append(f'Total samples ({total_samples}) below minimum ({MIN_SAMPLES_FOR_ML}). Predictions may be unreliable.')
        class_counts = y.value_counts()
        for cls, count in class_counts.items():
            if count < MIN_SAMPLES_PER_CLASS:
                warnings.append(f'Class {cls} has only {count} samples (minimum: {MIN_SAMPLES_PER_CLASS}). Model may not learn this class well.')
        if len(class_counts) == 2:
            minority_ratio = float(class_counts.min() / class_counts.sum())
            if minority_ratio < 0.1:
                warnings.append(f'Severe class imbalance detected ({minority_ratio:.1%} minority class).')
        for warning in warnings:
            logger.warning(warning)
        return warnings

    @staticmethod
    def _new_model(model_type: str, random_seed: int, params: Optional[dict] = None):
        params = dict(params or {})
        if model_type == 'random_forest':
            return RandomForestClassifier(**params, random_state=random_seed)
        if model_type == 'xgboost' and XGB_AVAILABLE:
            return XGBClassifier(**params, random_state=random_seed, eval_metric='logloss')
        if model_type == 'lightgbm' and LGBM_AVAILABLE:
            return LGBMClassifier(**params, random_state=random_seed, verbose=-1)
        raise MLEngineError(f'Unsupported or unavailable model type: {model_type}')

    def _evaluate_model_type(self, model_type: str, X: pd.DataFrame, y: pd.Series) -> float:
        model = self._new_model(model_type, self.random_seed)
        scores = cross_val_score(model, X, y, cv=3, scoring='f1')
        return float(np.mean(scores))

    def _evaluate_model_type_with_smote(self, model_type: str, X: pd.DataFrame, y: pd.Series) -> float:
        model = self._new_model(model_type, self.random_seed)
        if self.handle_imbalance and y.nunique() == 2:
            try:
                pipeline = ImbPipeline([
                    ('smote', SMOTE(random_state=self.random_seed)),
                    ('model', model),
                ])
                scores = cross_val_score(pipeline, X, y, cv=3, scoring='f1')
            except Exception as exc:
                logger.warning('SMOTE in CV failed (%s); evaluating without synthetic sampling', type(exc).__name__)
                scores = cross_val_score(model, X, y, cv=3, scoring='f1')
        else:
            scores = cross_val_score(model, X, y, cv=3, scoring='f1')
        return float(np.mean(scores))

    def _tune_and_train(self, model_type: str, X: pd.DataFrame, y: pd.Series) -> Any:
        """Legacy Optuna tuning without SMOTE, retained for compatibility."""
        def objective(trial):
            params = self._trial_params(model_type, trial)
            model = self._new_model(model_type, self.random_seed, params)
            return float(cross_val_score(model, X, y, cv=3, scoring='f1').mean())
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=self.random_seed))
        study.optimize(objective, n_trials=int(self.n_trials), show_progress_bar=False)
        model = self._new_model(model_type, self.random_seed, study.best_params)
        model.fit(X, y)
        return model

    @staticmethod
    def _trial_params(model_type: str, trial) -> dict:
        if model_type == 'random_forest':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            }
        if model_type == 'xgboost':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            }
        if model_type == 'lightgbm':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            }
        raise MLEngineError(f'Unsupported model type: {model_type}')

    def _tune_and_train_with_smote(self, model_type: str, X: pd.DataFrame, y: pd.Series) -> Any:
        def objective(trial):
            params = self._trial_params(model_type, trial)
            model = self._new_model(model_type, self.random_seed, params)
            if self.handle_imbalance and y.nunique() == 2:
                try:
                    pipeline = ImbPipeline([
                        ('smote', SMOTE(random_state=self.random_seed)),
                        ('model', model),
                    ])
                    return float(cross_val_score(pipeline, X, y, cv=3, scoring='f1').mean())
                except Exception:
                    pass
            return float(cross_val_score(model, X, y, cv=3, scoring='f1').mean())

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=self.random_seed))
        study.optimize(objective, n_trials=int(self.n_trials), show_progress_bar=False)
        model = self._new_model(model_type, self.random_seed, study.best_params)
        if self.handle_imbalance and y.nunique() == 2:
            try:
                X_resampled, y_resampled = SMOTE(random_state=self.random_seed).fit_resample(X, y)
                X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
                model.fit(X_resampled, y_resampled)
                return model
            except Exception as exc:
                logger.warning('Final SMOTE training failed (%s); fitting original training rows', type(exc).__name__)
        model.fit(X, y)
        return model

    def _prepare_shap(self, X: pd.DataFrame) -> None:
        self.shap_explainer = None
        self.shap_values = None
        try:
            import shap
            sample_size = min(self.shap_sample_size, len(X))
            if sample_size < 1 or self.model is None:
                return
            X_sample = X.sample(n=sample_size, random_state=self.random_seed)
            self.shap_explainer = shap.TreeExplainer(self.model)
            self.shap_values = self.shap_explainer.shap_values(X_sample)
        except ImportError:
            logger.warning('SHAP not available; diagnostic feature contributions are disabled')
        except Exception as exc:
            logger.warning('SHAP preparation failed: %s', type(exc).__name__)
            self.shap_explainer = None
            self.shap_values = None

    def _validated_scoring_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_trained or self.model is None:
            raise MLEngineError('Model not trained. Activate a passing model before scoring.')
        if not isinstance(X, pd.DataFrame):
            raise MLEngineError('Scoring requires a DataFrame')
        if not X.columns.is_unique:
            raise MLEngineError('Scoring requires unique feature columns')
        if self.feature_names and list(X.columns) != list(self.feature_names):
            raise MLEngineError('Scoring input must match the exact trained feature contract and order')
        if any(not pd.api.types.is_numeric_dtype(X[column]) for column in X):
            raise MLEngineError('Scoring requires finite numeric features')
        values = X.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise MLEngineError('Scoring requires finite numeric features')
        return X

    def predict_risk(self, X: pd.DataFrame) -> np.ndarray:
        """Internal scoring primitive used by governed aggregate runtime surfaces.

        Training requires both outcome classes. For compatibility with a previously
        serialized or externally supplied estimator, inference also handles a model
        whose ``classes_`` contains only 0 or only 1 without inventing the missing
        class probability: an all-0 classifier has attrition probability 0 and an
        all-1 classifier has attrition probability 1.
        """
        X = self._validated_scoring_frame(X)
        probabilities = np.asarray(self.model.predict_proba(X), dtype=float)
        classes = np.asarray(getattr(self.model, 'classes_', []))
        class_values = classes.tolist() if classes.ndim == 1 else []
        valid_classes = (
            classes.ndim == 1
            and 1 <= len(classes) <= 2
            and len(set(class_values)) == len(class_values)
            and set(class_values).issubset({0, 1})
        )
        if (
            not valid_classes
            or probabilities.shape != (len(X), len(classes))
            or not np.isfinite(probabilities).all()
            or (probabilities < 0).any() or (probabilities > 1).any()
            or not np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-7, atol=1e-9)
        ):
            raise MLEngineError('Expected finite binary attrition probabilities')
        if len(classes) == 1:
            return np.ones(len(X), dtype=float) if int(classes[0]) == 1 else np.zeros(len(X), dtype=float)
        positive = np.flatnonzero(classes == 1)
        if len(positive) != 1:
            raise MLEngineError('Expected exactly one positive attrition class')
        return probabilities[:, int(positive[0])]

    def predict_risk_with_confidence(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({
            'risk_score': self.predict_risk(X),
            'ci_lower': None,
            'ci_upper': None,
            'confidence_level': None,
            'confidence_category': 'Unavailable',
            'uncertainty_semantics': 'prediction_interval_not_estimated',
        })

    def get_risk_category(self, risk_score: float) -> str:
        if risk_score is None or not self._finite_real(risk_score) or not 0 <= float(risk_score) <= 1:
            return 'Unavailable'
        score = float(risk_score)
        if score >= self.risk_threshold_high:
            return 'High'
        if score >= self.risk_threshold_medium:
            return 'Medium'
        return 'Low'

    def get_risk_drivers(self, employee_idx: int, X: pd.DataFrame) -> list[dict]:
        """Internal diagnostic SHAP contributions; not employment-action advice."""
        X = self._validated_scoring_frame(X)
        if isinstance(employee_idx, (bool, np.bool_)) or not isinstance(employee_idx, (int, np.integer)):
            raise MLEngineError('Diagnostic row index must be an integer')
        if int(employee_idx) < 0 or int(employee_idx) >= len(X):
            raise MLEngineError('Diagnostic row index is out of range')
        if self.shap_explainer is None:
            raise MLEngineError('SHAP explainer not available and fallback to feature importances is disabled')
        try:
            employee_data = X.iloc[[int(employee_idx)]]
            shap_vals = self.shap_explainer.shap_values(employee_data)
            classes = np.asarray(getattr(self.model, 'classes_', []))
            positive = np.flatnonzero(classes == 1)
            if len(positive) != 1:
                raise MLEngineError('SHAP explanations require an identified attrition class')
            class_index = int(positive[0])
            if isinstance(shap_vals, list):
                if class_index >= len(shap_vals):
                    raise MLEngineError('SHAP output is missing the attrition class')
                values = np.asarray(shap_vals[class_index], dtype=float)[0]
            else:
                array = np.asarray(shap_vals, dtype=float)
                values = array[0, :, class_index] if array.ndim == 3 else array[0]
            if values.shape != (len(self.feature_names),) or not np.isfinite(values).all():
                raise MLEngineError('SHAP feature contributions do not align with the model')
            drivers = []
            for index, feature in enumerate(self.feature_names):
                raw_value = employee_data[feature].iloc[0]
                drivers.append({
                    'feature': feature,
                    'contribution': float(values[index]),
                    'output_units': 'raw_model_output',
                    'value': float(raw_value) if self._finite_real(raw_value) else None,
                    'abs_contribution': abs(float(values[index])),
                    'interpretation_boundary': 'diagnostic_model_contribution_not_causal_effect_or_action_recommendation',
                })
            return sorted(drivers, key=lambda item: item['abs_contribution'], reverse=True)
        except MLEngineError:
            raise
        except Exception as exc:
            logger.error('Error getting risk drivers: %s', type(exc).__name__)
            raise MLEngineError(f'Failed to calculate risk drivers: {type(exc).__name__}') from exc

    def get_recommendations(self, employee_id: str, risk_score: float, drivers: list[dict]) -> list[str]:
        """Individual employment-action recommendations are disabled by governance."""
        return []

    def get_feature_importance_summary(self) -> pd.DataFrame:
        if not self.is_trained or self.model is None:
            return pd.DataFrame(columns=['feature', 'importance'])
        if not hasattr(self.model, 'feature_importances_'):
            raise MLEngineError('Feature importance is unavailable for this model')
        importances = np.asarray(self.model.feature_importances_, dtype=float)
        if importances.shape != (len(self.feature_names),) or not np.isfinite(importances).all() or (importances < 0).any():
            raise MLEngineError('Feature importance output does not align with the trained feature contract')
        return pd.DataFrame({
            'feature': list(self.feature_names),
            'importance': importances.astype(float),
        }).sort_values('importance', ascending=False, kind='stable').reset_index(drop=True)


@st.cache_resource
def get_cached_model() -> MLEngine:
    return MLEngine()
