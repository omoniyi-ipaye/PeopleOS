"""Explicit model lifecycle with evaluation gate.

Training is a deliberate operation. Activation is blocked until deterministic
evaluation criteria pass. Heavy legacy ML dependencies are imported only when
an explicit training operation executes, keeping the control plane lightweight.
"""

from __future__ import annotations

from typing import Any, Dict

from .workspace import ModelState, ModelVersion, WorkspaceStore


class ModelEvaluationPolicy:
    """Deterministic minimum gate for candidate activation."""

    def __init__(self, min_auc: float = 0.60):
        self.min_auc = min_auc

    def evaluate(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        auc = metrics.get("roc_auc", metrics.get("auc", metrics.get("test_auc")))
        valid_auc = isinstance(auc, (int, float)) and 0.0 <= float(auc) <= 1.0
        passed = bool(valid_auc and float(auc) >= self.min_auc)
        return {
            "passed": passed,
            "checks": {
                "auc_present_and_valid": valid_auc,
                "auc_at_least_minimum": passed,
            },
            "thresholds": {"min_auc": self.min_auc},
            "observed": {"auc": float(auc) if valid_auc else None},
        }


class ModelLifecycleService:
    def __init__(self, store: WorkspaceStore, policy: ModelEvaluationPolicy | None = None):
        self.store = store
        self.policy = policy or ModelEvaluationPolicy()

    def train(self, workspace_id: str, dataset_id: str, features, target) -> ModelVersion:
        record = self.store.create_model(workspace_id=workspace_id, dataset_id=dataset_id)
        self.store.update_model(workspace_id, record.model_id, state=ModelState.TRAINING)
        try:
            # Import the legacy implementation only across the explicit training
            # boundary. Listing workspaces, checking health, or investigating a
            # question must never require the scientific training stack.
            from src.ml_engine import MLEngine

            engine = MLEngine()
            metrics = engine.train_model(features, target)
            self.store.update_model(workspace_id, record.model_id, state=ModelState.EVALUATING, metrics=metrics)
            evaluation = self.policy.evaluate(metrics)
            final_state = ModelState.CANDIDATE if evaluation["passed"] else ModelState.REJECTED
            return self.store.update_model(
                workspace_id,
                record.model_id,
                state=final_state,
                metrics=metrics,
                evaluation=evaluation,
            )
        except Exception:
            self.store.update_model(workspace_id, record.model_id, state=ModelState.FAILED)
            raise

    def activate(self, workspace_id: str, model_id: str) -> ModelVersion:
        return self.store.activate_model(workspace_id, model_id)
