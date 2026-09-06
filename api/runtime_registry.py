"""Workspace-scoped runtime registry for legacy analytics routes.

This transition adapter removes the process-global AppState sharing assumption
without changing the public contracts of the existing analytics endpoints.
FastAPI resolves the legacy `get_app_state` dependency through this registry.

The local workspace remains the default. Additional workspaces can opt in with
`X-PeopleOS-Workspace` once their dataset is loaded into an isolated runtime.
"""

from __future__ import annotations

import re
import threading
from typing import Dict

from fastapi import HTTPException, Request

from api.dependencies import AppState
from src.platform.workspace import WorkspaceStore

_WORKSPACE_RE = re.compile(r"^[A-Za-z0-9_-]{1,80}$")


class WorkspaceRuntimeState(AppState):
    """Independent AppState instance that bypasses the legacy singleton hook."""

    def __new__(cls, workspace_id: str):
        instance = object.__new__(cls)
        instance._initialized = False
        return instance

    def __init__(self, workspace_id: str):
        self.workspace_id = workspace_id
        super().__init__()


class WorkspaceRuntimeRegistry:
    """Process-local runtime instances keyed by stable workspace identity."""

    def __init__(self):
        self._lock = threading.RLock()
        self._states: Dict[str, WorkspaceRuntimeState] = {}
        self._workspace_store = WorkspaceStore()

    def get(self, workspace_id: str = "local") -> WorkspaceRuntimeState:
        if not _WORKSPACE_RE.fullmatch(workspace_id):
            raise ValueError("Invalid workspace identifier")
        self._workspace_store.ensure_workspace(workspace_id)
        with self._lock:
            state = self._states.get(workspace_id)
            if state is None:
                state = WorkspaceRuntimeState(workspace_id)
                self._states[workspace_id] = state
            return state

    def reset(self, workspace_id: str) -> None:
        with self._lock:
            self._states.pop(workspace_id, None)

    def list_loaded(self) -> list[str]:
        with self._lock:
            return sorted(self._states)


runtime_registry = WorkspaceRuntimeRegistry()


def get_workspace_state(request: Request) -> WorkspaceRuntimeState:
    """Resolve the current workspace runtime for FastAPI dependency injection."""
    workspace_id = request.headers.get("x-peopleos-workspace", "local").strip() or "local"
    try:
        return runtime_registry.get(workspace_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid workspace identifier") from exc


def get_local_state() -> WorkspaceRuntimeState:
    """Resolve the default local runtime outside a request dependency."""
    return runtime_registry.get("local")
