"""System health, fitness and bounded recovery checks for PeopleOS."""

from __future__ import annotations

import json
import hashlib
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List

from .workspace import WorkspaceStore, WorkspaceRecord, ModelState
from .runtime_lock import runtime_mutation


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
        recovery_required = False
        if registry_ok:
            records = self.store.list_workspaces()
            recovery_required = bool(self.store._read().get("recovery_required")) and not any(
                workspace.workspace_id == self.store.DEFAULT_WORKSPACE_ID and workspace.active_dataset_id
                for workspace in records
            )
            checks.append({"id": "recovery-dataset-reconciled", "healthy": not recovery_required})
            for workspace in records:
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
            "recovery_required": recovery_required,
            "checks": checks,
            "workspaces": workspaces,
            "adaptation_level": "L2-bounded-auto-heal",
            "autonomous_recovery_envelope": [
                "recreate missing local workspace metadata",
                "quarantine invalid registry bytes before rebuilding an empty metadata shell",
            ],
            "governed_only": [
                "change employee data",
                "activate a model",
                "change policy thresholds",
                "take employment action",
            ],
        }

    @runtime_mutation
    def recover(self) -> Dict[str, Any]:
        actions: List[str] = []
        quarantine = None
        repaired = False
        if not self._registry_valid():
            self.store.path.parent.mkdir(parents=True, exist_ok=True)
            if self.store.path.exists():
                # Preserve exact original bytes before any metadata replacement.
                # Exclusive creation prevents overwriting a prior quarantine.
                original = self.store.path.read_bytes()
                quarantine = self._quarantine_registry(original)
                if self.store.path.read_bytes() != original:
                    raise RuntimeError("Registry changed during recovery; original was preserved but no replacement was made")
                actions.append("preserved invalid registry for manual review")
            self.store._write({"schema_version": 1, "workspaces": [], "recovery_required": True, "recovery_quarantine": quarantine})
            repaired = True
            actions.append("recreated registry shell; dataset/model history requires manual reconciliation")
        # A failed ensure/check must propagate; never report a swallowed repair.
        self.store.ensure_workspace(self.store.DEFAULT_WORKSPACE_ID, "Local workspace")
        actions.append("ensured local workspace")
        health = self.check()
        return {
            "status": "metadata_reinitialized" if repaired else health["status"],
            "requires_review": repaired or health["status"] != "healthy",
            "actions": actions,
            "quarantine": quarantine,
            "health": health,
        }

    def _quarantine_registry(self, original: bytes) -> Dict[str, str]:
        backup = self.store.path.with_name(f"{self.store.path.name}.quarantine-{uuid.uuid4().hex}")
        # 0600 limits backup exposure even where the process umask is permissive.
        descriptor = os.open(backup, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, 'wb') as handle:
            handle.write(original)
            handle.flush()
            os.fsync(handle.fileno())
        if backup.read_bytes() != original:
            raise OSError("Registry quarantine verification failed; original registry retained")
        return {"filename": backup.name, "sha256": hashlib.sha256(original).hexdigest()}

    def _registry_valid(self) -> bool:
        try:
            payload = json.loads(Path(self.store.path).read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or payload.get("schema_version") != 1 or not isinstance(payload.get("workspaces"), list):
                return False
            records = [WorkspaceRecord.model_validate(item) for item in payload["workspaces"]]
            ids = [record.workspace_id for record in records]
            return len(ids) == len(set(ids))
        except Exception:
            return False
