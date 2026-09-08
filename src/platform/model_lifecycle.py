"""Explicit predictive model lifecycle with leakage-safe evaluation."""

from __future__ import annotations

from typing import Any, Dict

from .provenance import frame_fingerprint
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
        baseline = metrics.get('baseline_brier_score')
        baseline_pass = bool(valid_brier and isinstance(baseline, (int, float)) and 0 < baseline <= 1 and brier < baseline)
        ap, baseline_ap = metrics.get('average_precision'), metrics.get('baseline_average_precision')
        ap_pass = bool(isinstance(ap, (int, float)) and isinstance(baseline_ap, (int, float)) and 0 <= baseline_ap < ap <= 1)
        ece = metrics.get('calibration_error')
        calibration_pass = bool(isinstance(ece, (int, float)) and 0 <= ece <= .15)
        counts = metrics.get('test_class_counts')
        counts = counts if isinstance(counts, dict) else {}
        sample_pass = all(isinstance(counts.get(str(c)), int) and counts[str(c)] >= 10 for c in (0, 1))
        test_size = metrics.get('test_size')
        sample_pass = sample_pass and isinstance(test_size, int) and test_size >= 50
        fold_local = metrics.get('cv_preprocessing_fold_local') is True
        passed = bool(auc_pass and brier_pass and leakage_safe and baseline_pass and ap_pass and calibration_pass and sample_pass and fold_local)
        return {
            'passed': passed,
            'checks': {
                'auc_present_and_valid': valid_auc,
                'auc_at_least_minimum': auc_pass,
                'brier_present_and_valid': valid_brier,
                'brier_at_most_maximum': brier_pass,
                'preprocessing_fit_on_training_only': leakage_safe,
                'cv_preprocessing_fold_local': fold_local,
                'brier_beats_training_prevalence_baseline': baseline_pass,
                'average_precision_beats_prevalence': ap_pass,
                'weighted_calibration_error_at_most_015': calibration_pass,
                'holdout_at_least_50_and_10_per_class': sample_pass,
            },
            'thresholds': {'min_auc': self.min_auc, 'max_brier': self.max_brier, 'max_weighted_ece': .15, 'min_test_size': 50, 'min_test_per_class': 10},
            'scope': 'minimum_retrospective_gate_not_enterprise_or_future_prediction_certification',
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
        workspace = self.store.get_workspace(workspace_id)
        dataset = next((d for d in workspace.datasets if d.dataset_id == dataset_id), None)
        if dataset is None:
            raise KeyError('Unknown dataset')
        expected = dataset.quality.get('current_fingerprint')
        if expected and frame_fingerprint(raw_data) != expected:
            raise ValueError('Training input differs from the registered dataset snapshot')
        record = self.store.create_model(workspace_id=workspace_id, dataset_id=dataset_id)
        self.store.update_model(workspace_id, record.model_id, state=ModelState.TRAINING)
        try:
            from src.model_training import train_attrition_model
            artifact = train_attrition_model(raw_data)
            metrics = artifact.metrics
            metrics['training_current_fingerprint'] = frame_fingerprint(raw_data)
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
