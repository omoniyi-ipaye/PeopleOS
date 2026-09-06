"""Persistent operation state for ingest/training lifecycle work."""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class OperationJob(BaseModel):
    job_id: str
    workspace_id: str
    kind: str
    state: JobState = JobState.QUEUED
    idempotency_key: Optional[str] = None
    resource_id: Optional[str] = None
    error: Optional[str] = None
    created_at: str = Field(default_factory=_utcnow)
    updated_at: str = Field(default_factory=_utcnow)


class JobStore:
    def __init__(self, path: Optional[str] = None):
        self.path = Path(path or os.getenv("PEOPLEOS_JOB_REGISTRY", ".peopleos/jobs.json"))
        self._lock = threading.RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write({"schema_version": 1, "jobs": []})

    def _read(self) -> Dict[str, Any]:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(payload.get("jobs"), list):
                raise ValueError("invalid job registry")
            return payload
        except Exception:
            return {"schema_version": 1, "jobs": []}

    def _write(self, payload: Dict[str, Any]) -> None:
        with self._lock:
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            os.replace(tmp, self.path)

    def list(self, workspace_id: Optional[str] = None) -> List[OperationJob]:
        jobs = [OperationJob.model_validate(item) for item in self._read()["jobs"]]
        return [job for job in jobs if workspace_id is None or job.workspace_id == workspace_id]

    def create(self, *, workspace_id: str, kind: str, idempotency_key: Optional[str] = None) -> OperationJob:
        with self._lock:
            payload = self._read()
            if idempotency_key:
                for item in payload["jobs"]:
                    if item.get("workspace_id") == workspace_id and item.get("idempotency_key") == idempotency_key:
                        return OperationJob.model_validate(item)
            job = OperationJob(
                job_id=f"job_{uuid4().hex}",
                workspace_id=workspace_id,
                kind=kind,
                idempotency_key=idempotency_key,
            )
            payload["jobs"].append(job.model_dump(mode="json"))
            self._write(payload)
            return job

    def transition(self, job_id: str, state: JobState, *, resource_id: Optional[str] = None, error: Optional[str] = None) -> OperationJob:
        with self._lock:
            payload = self._read()
            for item in payload["jobs"]:
                if item["job_id"] == job_id:
                    item["state"] = state.value
                    item["updated_at"] = _utcnow()
                    if resource_id is not None:
                        item["resource_id"] = resource_id
                    if error is not None:
                        item["error"] = error[:500]
                    self._write(payload)
                    return OperationJob.model_validate(item)
        raise KeyError(f"Unknown job: {job_id}")

    def recover_interrupted(self) -> List[str]:
        """Fail closed for jobs left RUNNING after a process interruption."""
        recovered: List[str] = []
        for job in self.list():
            if job.state == JobState.RUNNING:
                self.transition(job.job_id, JobState.FAILED, error="Interrupted before completion; safe retry is allowed with the same idempotency key")
                recovered.append(job.job_id)
        return recovered
