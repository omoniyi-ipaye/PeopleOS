"""Deterministic dataset/model fitness monitoring for PeopleOS."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .workspace import ModelState, WorkspaceRecord


def _age_days(timestamp: Optional[str]) -> Optional[float]:
    if not timestamp:
        return None
    try:
        created = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - created).total_seconds() / 86400
    except ValueError:
        return None


class FitnessPolicy:
    def __init__(self, max_dataset_age_days: int = 90, max_model_age_days: int = 180, min_auc: float = 0.60):
        self.max_dataset_age_days = max_dataset_age_days
        self.max_model_age_days = max_model_age_days
        self.min_auc = min_auc

    def assess(self, workspace: WorkspaceRecord) -> Dict[str, Any]:
        dataset = next((d for d in workspace.datasets if d.dataset_id == workspace.active_dataset_id), None)
        model = next((m for m in workspace.models if m.model_id == workspace.active_model_id), None)

        dataset_age = _age_days(dataset.activated_at or dataset.created_at) if dataset else None
        model_age = _age_days(model.activated_at or model.created_at) if model else None
        auc = None
        if model:
            auc = model.metrics.get("roc_auc", model.metrics.get("auc", model.metrics.get("test_auc")))

        checks = {
            "active_dataset_present": dataset is not None,
            "dataset_fresh": dataset is not None and dataset_age is not None and dataset_age <= self.max_dataset_age_days,
            "active_model_consistent": model is None or model.state == ModelState.ACTIVE,
            "model_fresh": model is None or (model_age is not None and model_age <= self.max_model_age_days),
            "model_quality": model is None or (isinstance(auc, (int, float)) and float(auc) >= self.min_auc),
        }
        required = ["active_dataset_present", "dataset_fresh", "active_model_consistent", "model_fresh", "model_quality"]
        healthy = all(checks[key] for key in required)
        return {
            "status": "healthy" if healthy else "degraded",
            "checks": checks,
            "observed": {
                "dataset_age_days": round(dataset_age, 2) if dataset_age is not None else None,
                "model_age_days": round(model_age, 2) if model_age is not None else None,
                "model_auc": float(auc) if isinstance(auc, (int, float)) else None,
            },
            "thresholds": {
                "max_dataset_age_days": self.max_dataset_age_days,
                "max_model_age_days": self.max_model_age_days,
                "min_auc": self.min_auc,
            },
            "responses": {
                "stale_dataset": "ingest and validate a new dataset version; never silently replace data",
                "stale_model": "train and evaluate a candidate; never auto-activate it",
                "quality_regression": "mark model degraded and require governed review before replacement",
            },
        }
