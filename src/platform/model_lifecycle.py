"""Explicit predictive model lifecycle with leakage-safe evaluation."""

from __future__ import annotations

from typing import Any, Dict

from .workspace import ModelState, ModelVersion, WorkspaceStore


class ModelEvaluationPolicy:
    """Deterministic minimum gate; AUC alone never implies production fitness."""

    def __init__(self, min_auc: float = 0.60, max_brier: float = 0.30):
        self.min_auc = min_auc
        self.max_brier = max_brier

    def evaluate(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        auc = metrics.get('roc_auc', metrics.get('auc', metrics.get('test_auc')))
        brier = metrics.get('brier_score')
        leakage_safe = metrics.get('holdout_untouched_by_fit') is True
        valid_auc = isinstance(auc, (int, float)) and 0 <= float(auc) <= 1
        valid_brier = isinstance(brier, (int, float)) and 0 <= float(brier) <= 1
        auc_pass = bool(valid_auc and float(auc) >= self.min_auc)
        brier_pass = bool(valid_brier and float(brier) <= self.max_brier)
        passed = bool(auc_pass and brier_pass and leakage_safe)
        return {
            'passed': passed,
            'checks': {
                'auc_present_and_valid': valid_auc,
                'auc_at_least_minimum': auc_pass,
                'brier_present_and_valid': valid_brier,
                'brier_at_most_maximum': brier_pass,
                'preprocessing_fit_on_training_only': leakage_safe,
            },
            'thresholds': {'min_auc': self.min_auc, 'max_brier': self.max_brier},
            'observed': {
                'auc': float(auc) if valid_auc else None,
                'brier_score': float(brier) if valid_brier else None,
                'calibration_error': metrics.get('calibration_error'),
            },
        }


class ModelLifecycleService:
    """Creates immutable evaluated versions and retains runtime artifacts in-process.

    Durable model serialization remains an explicit platform gap; if a process
    restarts, metadata may survive but prediction capability must fail closed
    until a compatible artifact is restored/retrained.
    """

    _runtime_artifacts: dict[str, Any] = {}

    def __init__(self, store: WorkspaceStore, policy: ModelEvaluationPolicy | None = None):
        self.store = store
        self.policy = policy or ModelEvaluationPolicy()

    def train(self, workspace_id: str, dataset_id: str, raw_data) -> ModelVersion:
        record = self.store.create_model(workspace_id=workspace_id, dataset_id=dataset_id)
        self.store.update_model(workspace_id, record.model_id, state=ModelState.TRAINING)
        try:
            from src.model_training import train_attrition_model
            artifact = train_attrition_model(raw_data)
            metrics = artifact.metrics
            self._runtime_artifacts[record.model_id] = artifact
            self.store.update_model(workspace_id, record.model_id, state=ModelState.EVALUATING, metrics=metrics)
            evaluation = self.policy.evaluate(metrics)
            final_state = ModelState.CANDIDATE if evaluation['passed'] else ModelState.REJECTED
            return self.store.update_model(workspace_id, record.model_id, state=final_state, metrics=metrics, evaluation=evaluation)
        except Exception:
            self._runtime_artifacts.pop(record.model_id, None)
            self.store.update_model(workspace_id, record.model_id, state=ModelState.FAILED)
            raise

    def activate(self, workspace_id: str, model_id: str) -> ModelVersion:
        if model_id not in self._runtime_artifacts:
            raise ValueError('Model runtime artifact is unavailable in this process; retrain before activation')
        return self.store.activate_model(workspace_id, model_id)

    @classmethod
    def runtime_artifact(cls, model_id: str):
        return cls._runtime_artifacts.get(model_id)
