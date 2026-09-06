"""System health, fitness and bounded recovery checks for PeopleOS."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from .workspace import WorkspaceStore, ModelState


class SystemHealthMonitor:
    """Deterministic health monitor for the PeopleOS control plane.

    Recovery is intentionally bounded to metadata/configuration repair. It never
    changes employee data, model predictions, or consequential HR state.
    """

    def __init__(self, store: WorkspaceStore):
        self.store = store

    def check(self) -> Dict[str, Any]:
        checks: List[Dict[str, Any]] = []
        registry_ok = self._registry_valid()
        checks.append({"id": "registry-readable", "healthy": registry_ok})

        workspaces = []
        if registry_ok:
            for workspace in self.store.list_workspaces():
                active_dataset_ok = (
                    workspace.active_dataset_id is None
                    or any(d.dataset_id == workspace.active_dataset_id for d in workspace.datasets)
                )
                active_model_ok = (
                    workspace.active_model_id is None
                    or any(m.model_id == workspace.active_model_id and m.state == ModelState.ACTIVE for m in workspace.models)
                )
                workspaces.append({
                    "workspace_id": workspace.workspace_id,
                    "active_dataset_consistent": active_dataset_ok,
                    "active_model_consistent": active_model_ok,
                    "dataset_versions": len(workspace.datasets),
                    "model_versions": len(workspace.models),
                    "sessions": len(workspace.sessions),
                })
                checks.append({"id": f"workspace:{workspace.workspace_id}:dataset", "healthy": active_dataset_ok})
                checks.append({"id": f"workspace:{workspace.workspace_id}:model", "healthy": active_model_ok})

        healthy = all(item["healthy"] for item in checks)
        return {
            "status": "healthy" if healthy else "degraded",
            "checks": checks,
            "workspaces": workspaces,
            "adaptation_level": "L2-bounded-auto-heal",
            "autonomous_recovery_envelope": [
                "recreate missing local workspace metadata",
                "repair unreadable empty registry to schema shell",
            ],
            "governed_only": [
                "change employee data",
                "activate a model",
                "change policy thresholds",
                "take employment action",
            ],
        }

    def recover(self) -> Dict[str, Any]:
        actions: List[str] = []
        if not self._registry_valid():
            self.store.path.parent.mkdir(parents=True, exist_ok=True)
            self.store._write({"schema_version": 1, "workspaces": []})
            actions.append("recreated registry shell")
        try:
            self.store.ensure_workspace(self.store.DEFAULT_WORKSPACE_ID, "Local workspace")
            actions.append("ensured local workspace")
        except Exception:
            pass
        return {"actions": actions, "health": self.check()}

    def _registry_valid(self) -> bool:
        try:
            payload = json.loads(Path(self.store.path).read_text(encoding="utf-8"))
            return isinstance(payload, dict) and isinstance(payload.get("workspaces"), list)
        except Exception:
            return False
