"""Workspace, dataset, model, and investigation lifecycle registry.

This module stores only control-plane metadata. Raw employee data and model
artifacts remain behind their existing storage boundaries. The registry gives
PeopleOS stable identities and explicit state transitions so agent answers can
be traced to a workspace, dataset version, model version, and investigation.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from src.local_paths import get_peopleos_paths


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class DatasetState(str, Enum):
    REGISTERED = "registered"
    VALIDATED = "validated"
    ACTIVE = "active"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"


class ModelState(str, Enum):
    CREATED = "created"
    TRAINING = "training"
    EVALUATING = "evaluating"
    CANDIDATE = "candidate"
    ACTIVE = "active"
    REJECTED = "rejected"
    RETIRED = "retired"
    FAILED = "failed"


class SessionState(str, Enum):
    OPEN = "open"
    COMPLETE = "complete"
    FAILED = "failed"
    ARCHIVED = "archived"


class DatasetVersion(BaseModel):
    dataset_id: str
    workspace_id: str
    version: int = Field(ge=1)
    source_name: str
    content_hash: str
    row_count: int = Field(ge=0)
    columns: List[str] = Field(default_factory=list)
    state: DatasetState = DatasetState.REGISTERED
    quality: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=_utcnow)
    activated_at: Optional[str] = None


class ModelVersion(BaseModel):
    model_id: str
    workspace_id: str
    dataset_id: str
    model_family: str
    version: int = Field(ge=1)
    state: ModelState = ModelState.CREATED
    metrics: Dict[str, Any] = Field(default_factory=dict)
    evaluation: Dict[str, Any] = Field(default_factory=dict)
    artifact_ref: Optional[str] = None
    created_at: str = Field(default_factory=_utcnow)
    activated_at: Optional[str] = None


class InvestigationSession(BaseModel):
    session_id: str
    workspace_id: str
    dataset_id: Optional[str] = None
    model_id: Optional[str] = None
    state: SessionState = SessionState.OPEN
    question_hashes: List[str] = Field(default_factory=list)
    request_ids: List[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=_utcnow)
    updated_at: str = Field(default_factory=_utcnow)


class WorkspaceRecord(BaseModel):
    workspace_id: str
    name: str
    created_at: str = Field(default_factory=_utcnow)
    active_dataset_id: Optional[str] = None
    active_model_id: Optional[str] = None
    datasets: List[DatasetVersion] = Field(default_factory=list)
    models: List[ModelVersion] = Field(default_factory=list)
    sessions: List[InvestigationSession] = Field(default_factory=list)


class WorkspaceStore:
    """Small persistent control-plane registry with atomic JSON writes.

    The local workspace is created automatically. Writes are serialized and
    performed through os.replace so interruption cannot leave a partial file.
    By default the registry lives in the OS-native PeopleOS application-data
    directory; PEOPLEOS_WORKSPACE_REGISTRY can override it explicitly.
    """

    DEFAULT_WORKSPACE_ID = "local"

    def __init__(self, path: Optional[str] = None):
        default_path = os.getenv("PEOPLEOS_WORKSPACE_REGISTRY") or str(get_peopleos_paths().registry)
        self.path = Path(path or default_path)
        self._lock = threading.RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write({"schema_version": 1, "workspaces": []})
        self.ensure_workspace(self.DEFAULT_WORKSPACE_ID, "Local workspace")

    @staticmethod
    def hash_bytes(content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

    @staticmethod
    def hash_text(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def _read(self) -> Dict[str, Any]:
        with self._lock:
            try:
                with self.path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                if not isinstance(payload, dict) or payload.get("schema_version") != 1 or not isinstance(payload.get("workspaces"), list):
                    raise ValueError("Invalid registry schema")
                records = [WorkspaceRecord.model_validate(item) for item in payload["workspaces"]]
                identifiers = [record.workspace_id for record in records]
                if len(identifiers) != len(set(identifiers)):
                    raise ValueError("Duplicate workspace identities")
            except (FileNotFoundError, ValueError, UnicodeError) as exc:
                raise RuntimeError(
                    "Workspace registry is missing or invalid; original bytes were retained. "
                    "Restore a verified backup before startup, or use explicit bounded recovery "
                    "from an already-running instance."
                ) from exc
            return payload

    def _write(self, payload: Dict[str, Any]) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            with tmp.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, self.path)

    def list_workspaces(self) -> List[WorkspaceRecord]:
        payload = self._read()
        return [WorkspaceRecord.model_validate(item) for item in payload["workspaces"]]

    def get_workspace(self, workspace_id: str) -> WorkspaceRecord:
        for workspace in self.list_workspaces():
            if workspace.workspace_id == workspace_id:
                return workspace
        raise KeyError(f"Unknown workspace: {workspace_id}")

    def ensure_workspace(self, workspace_id: str, name: Optional[str] = None) -> WorkspaceRecord:
        with self._lock:
            payload = self._read()
            for item in payload["workspaces"]:
                if item["workspace_id"] == workspace_id:
                    return WorkspaceRecord.model_validate(item)
            workspace = WorkspaceRecord(workspace_id=workspace_id, name=name or workspace_id)
            payload["workspaces"].append(workspace.model_dump(mode="json"))
            self._write(payload)
            return workspace

    def _replace_workspace(self, workspace: WorkspaceRecord) -> WorkspaceRecord:
        with self._lock:
            payload = self._read()
            replaced = False
            for index, item in enumerate(payload["workspaces"]):
                if item["workspace_id"] == workspace.workspace_id:
                    payload["workspaces"][index] = workspace.model_dump(mode="json")
                    replaced = True
                    break
            if not replaced:
                payload["workspaces"].append(workspace.model_dump(mode="json"))
            self._write(payload)
            return workspace

    def register_dataset(
        self,
        *,
        workspace_id: str,
        source_name: str,
        content_hash: str,
        row_count: int,
        columns: List[str],
        quality: Optional[Dict[str, Any]] = None,
    ) -> DatasetVersion:
        workspace = self.ensure_workspace(workspace_id)
        next_version = 1 + max((dataset.version for dataset in workspace.datasets), default=0)
        dataset = DatasetVersion(
            dataset_id=f"ds_{uuid4().hex}",
            workspace_id=workspace_id,
            version=next_version,
            source_name=source_name,
            content_hash=content_hash,
            row_count=row_count,
            columns=columns,
            quality=quality or {},
            state=DatasetState.VALIDATED,
        )
        workspace.datasets.append(dataset)
        return self._replace_workspace(workspace).datasets[-1]

    def activate_dataset(self, workspace_id: str, dataset_id: str) -> DatasetVersion:
        workspace = self.get_workspace(workspace_id)
        selected: Optional[DatasetVersion] = None
        for dataset in workspace.datasets:
            if dataset.dataset_id == dataset_id:
                dataset.state = DatasetState.ACTIVE
                dataset.activated_at = _utcnow()
                selected = dataset
            elif dataset.state == DatasetState.ACTIVE:
                dataset.state = DatasetState.SUPERSEDED
        if selected is None:
            raise KeyError(f"Unknown dataset: {dataset_id}")
        for model in workspace.models:
            if model.state == ModelState.ACTIVE:
                model.state = ModelState.RETIRED
        workspace.active_model_id = None
        workspace.active_dataset_id = selected.dataset_id
        self._replace_workspace(workspace)
        return selected

    def create_model(
        self,
        *,
        workspace_id: str,
        dataset_id: str,
        model_family: str = "attrition-risk",
    ) -> ModelVersion:
        workspace = self.get_workspace(workspace_id)
        if not any(dataset.dataset_id == dataset_id for dataset in workspace.datasets):
            raise KeyError(f"Unknown dataset: {dataset_id}")
        family_versions = [m.version for m in workspace.models if m.model_family == model_family]
        model = ModelVersion(
            model_id=f"model_{uuid4().hex}",
            workspace_id=workspace_id,
            dataset_id=dataset_id,
            model_family=model_family,
            version=1 + max(family_versions, default=0),
        )
        workspace.models.append(model)
        self._replace_workspace(workspace)
        return model

    def update_model(
        self,
        workspace_id: str,
        model_id: str,
        *,
        state: Optional[ModelState] = None,
        metrics: Optional[Dict[str, Any]] = None,
        evaluation: Optional[Dict[str, Any]] = None,
        artifact_ref: Optional[str] = None,
    ) -> ModelVersion:
        workspace = self.get_workspace(workspace_id)
        selected: Optional[ModelVersion] = None
        for model in workspace.models:
            if model.model_id == model_id:
                if state is not None:
                    model.state = state
                if metrics is not None:
                    model.metrics = metrics
                if evaluation is not None:
                    model.evaluation = evaluation
                if artifact_ref is not None:
                    model.artifact_ref = artifact_ref
                selected = model
                break
        if selected is None:
            raise KeyError(f"Unknown model: {model_id}")
        self._replace_workspace(workspace)
        return selected

    def activate_model(self, workspace_id: str, model_id: str) -> ModelVersion:
        workspace = self.get_workspace(workspace_id)
        selected: Optional[ModelVersion] = None
        for model in workspace.models:
            if model.model_id == model_id:
                if model.dataset_id != workspace.active_dataset_id:
                    raise ValueError('Model dataset differs from the active dataset')
                if model.state != ModelState.CANDIDATE:
                    raise ValueError("Only an evaluated candidate model can be activated")
                model.state = ModelState.ACTIVE
                model.activated_at = _utcnow()
                selected = model
            elif model.state == ModelState.ACTIVE:
                model.state = ModelState.RETIRED
        if selected is None:
            raise KeyError(f"Unknown model: {model_id}")
        workspace.active_model_id = selected.model_id
        self._replace_workspace(workspace)
        return selected

    def open_session(
        self,
        *,
        workspace_id: str,
        dataset_id: Optional[str] = None,
        model_id: Optional[str] = None,
    ) -> InvestigationSession:
        workspace = self.ensure_workspace(workspace_id)
        selected_dataset = dataset_id or workspace.active_dataset_id
        selected_model = model_id or workspace.active_model_id
        if selected_dataset and not any(d.dataset_id == selected_dataset for d in workspace.datasets):
            raise KeyError('Unknown investigation dataset')
        if selected_model:
            model = next((m for m in workspace.models if m.model_id == selected_model), None)
            if model is None:
                raise KeyError('Unknown investigation model')
            if model.dataset_id != selected_dataset:
                raise ValueError('Investigation model and dataset are incompatible')
        session = InvestigationSession(
            session_id=f"session_{uuid4().hex}",
            workspace_id=workspace_id,
            dataset_id=dataset_id or workspace.active_dataset_id,
            model_id=model_id or workspace.active_model_id,
        )
        workspace.sessions.append(session)
        self._replace_workspace(workspace)
        return session

    def record_request(self, workspace_id: str, session_id: str, request_id: str, question: str) -> InvestigationSession:
        workspace = self.get_workspace(workspace_id)
        selected: Optional[InvestigationSession] = None
        for session in workspace.sessions:
            if session.session_id == session_id:
                session.request_ids.append(request_id)
                session.question_hashes.append(self.hash_text(question))
                session.updated_at = _utcnow()
                selected = session
                break
        if selected is None:
            raise KeyError(f"Unknown session: {session_id}")
        self._replace_workspace(workspace)
        return selected

    def close_session(self, workspace_id: str, session_id: str, *, failed: bool = False) -> InvestigationSession:
        workspace = self.get_workspace(workspace_id)
        selected: Optional[InvestigationSession] = None
        for session in workspace.sessions:
            if session.session_id == session_id:
                session.state = SessionState.FAILED if failed else SessionState.COMPLETE
                session.updated_at = _utcnow()
                selected = session
                break
        if selected is None:
            raise KeyError(f"Unknown session: {session_id}")
        self._replace_workspace(workspace)
        return selected
