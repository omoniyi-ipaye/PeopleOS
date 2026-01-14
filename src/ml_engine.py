"""
Machine Learning Engine module for PeopleOS.

Provides attrition risk prediction using Random Forest,
SHAP-based explanations, and rule-based recommendations.
"""

from typing import Any, Optional

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
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

# Minimum sample size requirements for reliable ML
MIN_SAMPLES_FOR_ML = 50
MIN_SAMPLES_PER_CLASS = 10
from src.utils import load_config
from src.preprocessor import Preprocessor

logger = get_logger('ml_engine')


# Rule-based recommendation templates
RECOMMENDATION_RULES = [
    {
        'condition': lambda risk, drivers: risk > 0.75 and 'Salary' in [d['feature'] for d in drivers[:3]],
        'recommendation': "Review compensation band against market rate"
    },
    {
        'condition': lambda risk, drivers: risk > 0.75 and any(d['feature'] == 'Tenure' and d['value'] < 1 for d in drivers),
        'recommendation': "Schedule 30-day check-in with manager"
    },
    {
        'condition': lambda risk, drivers: risk > 0.50 and 'LastRating' in [d['feature'] for d in drivers[:3]],
        'recommendation': "Review recent performance feedback quality"
    },
    {
        'condition': lambda risk, drivers: risk > 0.50 and 'Dept' in [d['feature'] for d in drivers[:3]],
        'recommendation': "Assess team culture and management effectiveness"
    },
    {
        'condition': lambda risk, drivers: risk > 0.75,
        'recommendation': "Conduct stay interview to understand concerns"
    },
    {
        'condition': lambda risk, drivers: risk > 0.50 and risk <= 0.75,
        'recommendation': "Include in retention program and monitor engagement"
    }
]


class MLEngineError(Exception):
    """Custom exception for ML engine errors."""
    pass


class MLEngine:
    """
    Machine learning engine for attrition risk prediction.
    
    Uses Random Forest classifier with SHAP explanations.
    """
    
    def __init__(self):
        """Initialize the ML Engine with configuration."""
        self.config = load_config()
        self.ml_config = self.config.get('ml', {})
        self.random_seed = self.ml_config.get('random_seed', 42)
        self.shap_sample_size = self.ml_config.get('shap_sample_size', 100)
        self.risk_threshold_high = self.ml_config.get('risk_threshold_high', 0.75)
        self.risk_threshold_medium = self.ml_config.get('risk_threshold_medium', 0.50)
        self.test_split_ratio = self.ml_config.get('test_split_ratio', 0.2)
        
        self.model: Optional[Any] = None
        self.best_model_name: str = ""
        self.feature_names: list[str] = []
        self.shap_explainer: Any = None
        self.shap_values: Optional[np.ndarray] = None
        self.is_trained = False
        self.handle_imbalance = self.ml_config.get('handle_imbalance', True)
        self.models_to_compare = self.ml_config.get('models', ['random_forest', 'xgboost', 'lightgbm'])
        self.n_trials = self.ml_config.get('optuna_trials', 10)
        self.preprocessor = Preprocessor()
    
    def train(self, df: pd.DataFrame) -> dict:
        """
        High-level training method that handles preprocessing and model selection.
        
        Args:
            df: Raw DataFrame with employee data.
            
        Returns:
            Dictionary with training metrics.
        """
        logger.info(f"Starting ML training pipeline for {len(df)} employees")
        
        # 1. Preprocess
        target_col = 'Attrition' if 'Attrition' in df.columns else 'attrition'
        processed_df, metadata = self.preprocessor.fit_transform(df, target_column=target_col)
        
        # 2. Split into X and y
        feature_cols = self.preprocessor.numeric_columns + self.preprocessor.categorical_columns
        X = processed_df[feature_cols]
        
        # DIAGNOSTIC: Check for non-numeric columns
        non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            logger.error(f"CRITICAL: Non-numeric columns in training set: {non_numeric}. Dropping them.")
            X = X.drop(columns=non_numeric)
        
        if target_col in processed_df.columns:
            y = processed_df[target_col]
        else:
            y = pd.Series([0] * len(processed_df))

        # DIAGNOSTIC: Check y for non-numeric data
        if not pd.api.types.is_numeric_dtype(y):
            logger.error(f"CRITICAL: Target variable y is non-numeric (type: {y.dtype}). Dropping non-numeric rows.")
            logger.error(f"Sample value for y: {y.iloc[0]}")
            # Try to force numeric
            y = pd.to_numeric(y, errors='coerce').fillna(0).astype(int)
            
        # 3. Train
        metrics = self.train_model(X, y)
        self.is_trained = True
        return metrics

    def predict(self, df: pd.DataFrame) -> list[dict]:
        """
        High-level prediction method that handles preprocessing and risk scoring.
        
        Args:
            df: Raw DataFrame with employee data.
            
        Returns:
            List of dictionaries with risk scores and categories.
        """
        if not self.is_trained or self.model is None:
            return []
            
        # 1. Transform raw data
        processed_df = self.preprocessor.transform(df)
        
        # 2. Select feature columns
        feature_cols = self.preprocessor.numeric_columns + self.preprocessor.categorical_columns
        X = processed_df[feature_cols]
        
        # 3. Predict
        scores = self.predict_risk(X)
        
        results = []
        for i, score in enumerate(scores):
            results.append({
                'risk_score': float(score),
                'risk_category': self.get_risk_category(score)
            })
            
        return results
    
    def train_model(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """
        Train the attrition prediction model using automated selection and return metrics.

        Includes sample size validation and proper CV-SMOTE integration to prevent data leakage.
        """
        try:
            self.feature_names = list(X.columns)

            # CRITICAL: Sample size validation
            sample_size_warnings = self._validate_sample_size(X, y)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.test_split_ratio,
                random_state=self.random_seed,
                stratify=y if len(y.unique()) > 1 else None
            )

            # Store original training data for proper CV (SMOTE applied inside CV folds)
            X_train_original = X_train.copy()
            y_train_original = y_train.copy()

            # Model Selection via Cross-Validation
            # Using ImbPipeline to apply SMOTE inside CV folds (prevents data leakage)
            candidate_models = ['random_forest']
            if XGB_AVAILABLE:
                candidate_models.append('xgboost')
            if LGBM_AVAILABLE:
                candidate_models.append('lightgbm')

            logger.info(f"Evaluating model candidates: {candidate_models}")
            best_model_type = 'random_forest'  # Default to RF
            best_f1 = -1.0

            results = {}

            # SINGLE-CLASS BYPASS: Skip model selection if only 1 class in training set
            if len(y_train_original.unique()) < 2:
                logger.warning("Only 1 class in y_train. Skipping model selection and tuning.")
                self.model = RandomForestClassifier(random_state=self.random_seed)
                self.model.fit(X_train_original, y_train_original)
                self.best_model_name = 'random_forest'
            else:
                for m_type in candidate_models:
                    # Use proper CV with SMOTE inside folds
                    score = self._evaluate_model_type_with_smote(m_type, X_train_original, y_train_original)
                    results[m_type] = score
                    if score > best_f1:
                        best_f1 = score
                        best_model_type = m_type

                logger.info(f"Model selection results: {results}")
                logger.info(f"Selected best model: {best_model_type}")

                # Tune and train the best model (SMOTE applied only once on final training)
                self.model = self._tune_and_train_with_smote(best_model_type, X_train_original, y_train_original)
                self.best_model_name = best_model_type


            # Calculate metrics
            y_pred = self.model.predict(X_test)
            y_proba = self.model.predict_proba(X_test)[:, 1] if len(self.model.classes_) == 2 else None
            
            metrics = {
                'accuracy': float(accuracy_score(y_test, y_pred)),
                'precision': float(precision_score(y_test, y_pred, zero_division=0)),
                'recall': float(recall_score(y_test, y_pred, zero_division=0)),
                'f1': float(f1_score(y_test, y_pred, zero_division=0)),
                'best_model': best_model_type,
                'train_size': len(X_train),
                'test_size': len(X_test)
            }
            
            if y_proba is not None and len(y.unique()) == 2:
                try:
                    metrics['roc_auc'] = float(roc_auc_score(y_test, y_proba))
                    metrics['brier_score'] = float(brier_score_loss(y_test, y_proba))
                    
                    # 5. Check Calibration (with sample size consideration)
                    prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=5)
                    calibration_error = np.mean(np.abs(prob_true - prob_pred))
                    
                    # Only flag calibration error if we have sufficient test data
                    if calibration_error > 0.15 and len(y_test) > 100:
                        metrics['warnings'].append(
                            f"Model calibration is poor (>15% error). Risk probabilities may be inaccurate."
                        )
                    elif calibration_error > 0.15:
                         # Just log, don't warn user if data is too small to be sure
                        logger.info(f"Calibration error {calibration_error:.2f} but sample size {len(y_test)} too small to warn.")
                        
                    metrics['calibration_error'] = float(calibration_error)
                except ValueError:
                    metrics['roc_auc'] = None
            
            # Feature importances
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                metrics['feature_importances'] = dict(zip(self.feature_names, importances.tolist()))

            # Add sample size warnings to metrics
            if sample_size_warnings:
                metrics['warnings'] = sample_size_warnings
                metrics['reliability'] = 'Low' if len(sample_size_warnings) > 1 else 'Medium'
            else:
                metrics['reliability'] = 'High'

            self.is_trained = True
            logger.info(f"Model trained. Best: {best_model_type}, F1: {metrics['f1']:.3f}, Reliability: {metrics['reliability']}")
            
            # Initialize SHAP
            self._prepare_shap(X)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Model training failed: {type(e).__name__}: {str(e)}")
            raise MLEngineError(f"Model training failed: {str(e)}")

    def _validate_sample_size(self, X: pd.DataFrame, y: pd.Series) -> list:
        """
        Validate that sample sizes are sufficient for reliable ML predictions.

        Returns list of warning messages if sample sizes are insufficient.
        """
        warnings = []

        total_samples = len(X)
        if total_samples < MIN_SAMPLES_FOR_ML:
            warnings.append(
                f"Total samples ({total_samples}) below minimum ({MIN_SAMPLES_FOR_ML}). "
                "Predictions may be unreliable."
            )

        # Check class distribution
        class_counts = y.value_counts()
        for cls, count in class_counts.items():
            if count < MIN_SAMPLES_PER_CLASS:
                warnings.append(
                    f"Class {cls} has only {count} samples (minimum: {MIN_SAMPLES_PER_CLASS}). "
                    "Model may not learn this class well."
                )

        # Check for extreme imbalance
        if len(class_counts) == 2:
            minority_ratio = class_counts.min() / class_counts.sum()
            if minority_ratio < 0.1:
                warnings.append(
                    f"Severe class imbalance detected ({minority_ratio:.1%} minority class). "
                    "Consider collecting more data or using specialized techniques."
                )

        if warnings:
            for w in warnings:
                logger.warning(w)

        return warnings

    def _evaluate_model_type(self, model_type: str, X: pd.DataFrame, y: pd.Series) -> float:
        """Quick evaluation of a model type using CV (deprecated - use _evaluate_model_type_with_smote)."""
        if model_type == 'random_forest':
            model = RandomForestClassifier(random_state=self.random_seed)
        elif model_type == 'xgboost':
            model = XGBClassifier(random_state=self.random_seed, use_label_encoder=False, eval_metric='logloss')
        elif model_type == 'lightgbm':
            model = LGBMClassifier(random_state=self.random_seed, verbose=-1)
        else:
            return 0.0

        scores = cross_val_score(model, X, y, cv=3, scoring='f1')
        return float(np.mean(scores))

    def _evaluate_model_type_with_smote(self, model_type: str, X: pd.DataFrame, y: pd.Series) -> float:
        """
        Evaluate a model type using CV with SMOTE applied inside each fold.

        This prevents data leakage by ensuring synthetic samples are only
        generated from training data within each fold.
        """
        if model_type == 'random_forest':
            model = RandomForestClassifier(random_state=self.random_seed)
        elif model_type == 'xgboost':
            model = XGBClassifier(random_state=self.random_seed, use_label_encoder=False, eval_metric='logloss')
        elif model_type == 'lightgbm':
            model = LGBMClassifier(random_state=self.random_seed, verbose=-1)
        else:
            return 0.0

        # Use ImbPipeline to apply SMOTE inside each CV fold
        if self.handle_imbalance and len(y.unique()) == 2:
            try:
                pipeline = ImbPipeline([
                    ('smote', SMOTE(random_state=self.random_seed)),
                    ('model', model)
                ])
                scores = cross_val_score(pipeline, X, y, cv=3, scoring='f1')
            except Exception as e:
                logger.warning(f"SMOTE in CV failed: {str(e)}. Using model without SMOTE.")
                scores = cross_val_score(model, X, y, cv=3, scoring='f1')
        else:
            scores = cross_val_score(model, X, y, cv=3, scoring='f1')

        return float(np.mean(scores))

    def _tune_and_train(self, model_type: str, X: pd.DataFrame, y: pd.Series) -> Any:
        """Tune hyperparameters using Optuna and return the best model."""
        
        def objective(trial):
            if model_type == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10)
                }
                model = RandomForestClassifier(**params, random_state=self.random_seed)
            elif model_type == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3)
                }
                model = XGBClassifier(**params, random_state=self.random_seed, eval_metric='logloss')
            elif model_type == 'lightgbm':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 100)
                }
                model = LGBMClassifier(**params, random_state=self.random_seed, verbose=-1)
            else:
                return 0.0
            
            score = cross_val_score(model, X, y, cv=3, scoring='f1').mean()
            return score

        # Optimize
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)
        best_params = study.best_params
        
        # Train final model with best params
        if model_type == 'random_forest':
            model = RandomForestClassifier(**best_params, random_state=self.random_seed)
        elif model_type == 'xgboost':
            model = XGBClassifier(**best_params, random_state=self.random_seed, eval_metric='logloss')
        elif model_type == 'lightgbm':
            model = LGBMClassifier(**best_params, random_state=self.random_seed, verbose=-1)
        
        model.fit(X, y)
        return model

    def _tune_and_train_with_smote(self, model_type: str, X: pd.DataFrame, y: pd.Series) -> Any:
        """
        Tune hyperparameters using Optuna with SMOTE inside CV folds, then train final model.

        This ensures no data leakage during hyperparameter optimization.
        """

        def objective(trial):
            if model_type == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10)
                }
                model = RandomForestClassifier(**params, random_state=self.random_seed)
            elif model_type == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3)
                }
                model = XGBClassifier(**params, random_state=self.random_seed, eval_metric='logloss')
            elif model_type == 'lightgbm':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 100)
                }
                model = LGBMClassifier(**params, random_state=self.random_seed, verbose=-1)
            else:
                return 0.0

            # Use ImbPipeline for SMOTE inside CV folds
            if self.handle_imbalance and len(y.unique()) == 2:
                try:
                    pipeline = ImbPipeline([
                        ('smote', SMOTE(random_state=self.random_seed)),
                        ('model', model)
                    ])
                    score = cross_val_score(pipeline, X, y, cv=3, scoring='f1').mean()
                except Exception:
                    score = cross_val_score(model, X, y, cv=3, scoring='f1').mean()
            else:
                score = cross_val_score(model, X, y, cv=3, scoring='f1').mean()

            return score

        # Optimize with Optuna (suppress output, fixed seed for consistency)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        sampler = TPESampler(seed=self.random_seed)
        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        best_params = study.best_params

        # Train final model with best params
        if model_type == 'random_forest':
            model = RandomForestClassifier(**best_params, random_state=self.random_seed)
        elif model_type == 'xgboost':
            model = XGBClassifier(**best_params, random_state=self.random_seed, eval_metric='logloss')
        elif model_type == 'lightgbm':
            model = LGBMClassifier(**best_params, random_state=self.random_seed, verbose=-1)

        # Apply SMOTE once on full training data for final model
        if self.handle_imbalance and len(y.unique()) == 2:
            try:
                smote = SMOTE(random_state=self.random_seed)
                X_resampled_arr, y_resampled = smote.fit_resample(X, y)
                # Convert back to DataFrame to preserve feature names and avoid sklearn warnings
                X_resampled = pd.DataFrame(X_resampled_arr, columns=X.columns)
                model.fit(X_resampled, y_resampled)
                logger.info("Final model trained with SMOTE-resampled data")
            except Exception as e:
                logger.warning(f"SMOTE failed for final model: {str(e)}. Training without SMOTE.")
                model.fit(X, y)
        else:
            model.fit(X, y)

        return model

    def _prepare_shap(self, X: pd.DataFrame) -> None:
        """
        Prepare SHAP explainer with sampled data.
        
        Args:
            X: Feature DataFrame.
        """
        try:
            import shap
            
            # Sample data for SHAP to improve performance
            sample_size = min(self.shap_sample_size, len(X))
            X_sample = X.sample(n=sample_size, random_state=self.random_seed)
            
            # Use TreeExplainer for Random Forest
            self.shap_explainer = shap.TreeExplainer(self.model)
            self.shap_values = self.shap_explainer.shap_values(X_sample)
            
            logger.info(f"SHAP explainer prepared with {sample_size} samples")
            
        except ImportError:
            logger.warning("SHAP not available, explanations will use feature importances")
            self.shap_explainer = None
        except Exception as e:
            logger.warning(f"SHAP preparation failed: {str(e)}")
            self.shap_explainer = None
    
    def predict_risk(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return risk probabilities for each employee.

        Args:
            X: Feature DataFrame.

        Returns:
            Array of risk probabilities.
        """
        if not self.is_trained or self.model is None:
            raise MLEngineError("Model not trained. Call train_model first.")

        # Get probability of positive class (attrition = 1)
        probabilities = self.model.predict_proba(X)

        if len(self.model.classes_) == 2:
            # Return probability of attrition (class 1)
            risk_scores = probabilities[:, 1]
        else:
            # If multi-class, return max probability
            risk_scores = probabilities.max(axis=1)

        return risk_scores

    def predict_risk_with_confidence(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Return risk probabilities with confidence intervals for each employee.

        Uses bootstrap aggregation from Random Forest trees to estimate uncertainty.

        Args:
            X: Feature DataFrame.

        Returns:
            DataFrame with risk_score, confidence_lower, confidence_upper, confidence_level.
        """
        if not self.is_trained or self.model is None:
            raise MLEngineError("Model not trained. Call train_model first.")

        risk_scores = self.predict_risk(X)

        # For tree-based models, use individual tree predictions for confidence intervals
        if hasattr(self.model, 'estimators_'):
            # Get predictions from each tree
            tree_predictions = []
            for tree in self.model.estimators_:
                if hasattr(tree, 'predict_proba'):
                    proba = tree.predict_proba(X)
                    if proba.shape[1] == 2:
                        tree_predictions.append(proba[:, 1])
                    else:
                        tree_predictions.append(proba.max(axis=1))

            if tree_predictions:
                tree_predictions = np.array(tree_predictions)
                # Calculate 90% confidence interval
                lower = np.percentile(tree_predictions, 5, axis=0)
                upper = np.percentile(tree_predictions, 95, axis=0)
                std_dev = np.std(tree_predictions, axis=0)

                # Confidence level based on agreement among trees
                confidence_level = 1 - (std_dev / 0.5)  # Normalize: max std for binary is ~0.5
                confidence_level = np.clip(confidence_level, 0, 1)
            else:
                # Fallback if no tree predictions
                lower = risk_scores * 0.8
                upper = np.minimum(risk_scores * 1.2, 1.0)
                confidence_level = np.full_like(risk_scores, 0.5)
        else:
            # Fallback for non-ensemble models (like XGBoost, LogisticRegression)
            # Use a slightly wider interval for non-confidence-aware models
            lower = np.maximum(risk_scores - 0.1, 0.0)
            upper = np.minimum(risk_scores + 0.1, 1.0)
            confidence_level = np.full_like(risk_scores, 0.6)  # Moderate confidence

        result = pd.DataFrame({
            'risk_score': risk_scores,
            'ci_lower': np.round(lower, 3),
            'ci_upper': np.round(upper, 3),
            'confidence_level': np.round(confidence_level, 2)
        })

        # Add confidence category
        result['confidence_category'] = result['confidence_level'].apply(
            lambda x: 'High' if x >= 0.7 else ('Medium' if x >= 0.4 else 'Low')
        )

        return result
    
    def get_risk_category(self, risk_score: float) -> str:
        """
        Categorize risk score into High/Medium/Low.
        
        Args:
            risk_score: Risk probability.
            
        Returns:
            Risk category string.
        """
        if risk_score >= self.risk_threshold_high:
            return "High"
        elif risk_score >= self.risk_threshold_medium:
            return "Medium"
        else:
            return "Low"
    
    def get_risk_drivers(self, employee_idx: int, X: pd.DataFrame) -> list[dict]:
        """
        Return SHAP-based feature contributions for an employee.
        
        Args:
            employee_idx: Index of the employee in the DataFrame.
            X: Feature DataFrame.
            
        Returns:
            List of dictionaries with feature contributions.
        """
        if not self.is_trained or self.model is None:
            return []
        
        drivers = []
        
        try:
            if self.shap_explainer is not None:
                
                # Get SHAP values for this employee
                employee_data = X.iloc[[employee_idx]]
                shap_vals = self.shap_explainer.shap_values(employee_data)
                
                # Handle both binary and multi-class
                if isinstance(shap_vals, list):
                    # Binary classification - use class 1 (attrition)
                    vals = shap_vals[1][0] if len(shap_vals) > 1 else shap_vals[0][0]
                else:
                    vals = shap_vals[0]
                
                # Create driver list
                for i, feature in enumerate(self.feature_names):
                    drivers.append({
                        'feature': feature,
                        'contribution': float(vals[i]),
                        'value': float(employee_data[feature].iloc[0]) if feature in employee_data.columns else None,
                        'abs_contribution': abs(float(vals[i]))
                    })
                
            else:
                raise MLEngineError("SHAP explainer not available and fallback to feature importances is disabled")
            
            # Sort by absolute contribution
            drivers.sort(key=lambda x: x['abs_contribution'], reverse=True)
            
        except Exception as e:
            logger.error(f"Error getting risk drivers: {str(e)}")
            raise MLEngineError(f"Failed to calculate risk drivers: {str(e)}")
        
        return drivers
    
    def get_recommendations(self, employee_id: str, risk_score: float, drivers: list[dict]) -> list[str]:
        """
        Return rule-based recommendations.
        
        Args:
            employee_id: Employee identifier (for logging only).
            risk_score: Risk probability.
            drivers: List of risk drivers from get_risk_drivers.
            
        Returns:
            List of recommendation strings.
        """
        recommendations = []
        
        for rule in RECOMMENDATION_RULES:
            try:
                if rule['condition'](risk_score, drivers):
                    if rule['recommendation'] not in recommendations:
                        recommendations.append(rule['recommendation'])
            except Exception:
                continue
        
        # Limit to 3 recommendations
        recommendations = recommendations[:3]
        
        # If no recommendations matched, provide a generic one
        if not recommendations and risk_score > self.risk_threshold_medium:
            recommendations.append("Monitor employee engagement and schedule follow-up")
        
        logger.info(f"Generated {len(recommendations)} recommendations for employee")
        return recommendations
    
    def get_feature_importance_summary(self) -> pd.DataFrame:
        """
        Get a summary of feature importances.
        
        Returns:
            DataFrame with feature importance rankings.
        """
        if not self.is_trained or self.model is None:
            return pd.DataFrame()
        
        importances = self.model.feature_importances_
        
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return df


@st.cache_resource
def get_cached_model() -> MLEngine:
    """
    Get a cached ML engine instance.
    
    Returns:
        Cached MLEngine instance.
    """
    return MLEngine()
