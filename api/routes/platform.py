"""Workspace, dataset/model lifecycle, session, monitoring and health API."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from api.authorization import actor_from_request, permissions_for_role, require_permission
from api.dependencies import AppState, get_app_state
from src.platform.health import SystemHealthMonitor
from src.platform.jobs import JobState, JobStore
from src.platform.model_lifecycle import ModelLifecycleService
from src.platform.monitoring import FitnessPolicy
from src.platform.workspace import WorkspaceStore

router = APIRouter(prefix="/api/platform", tags=["platform"])
_store = WorkspaceStore()
_jobs = JobStore()


class WorkspaceRequest(BaseModel):
    workspace_id: str = Field(min_length=1, max_length=80, pattern=r"^[A-Za-z0-9_-]+$")
    name: str = Field(min_length=1, max_length=120)


class SessionRequest(BaseModel):
    workspace_id: str = "local"
    dataset_id: Optional[str] = None
    model_id: Optional[str] = None


@router.get("/me")
async def who_am_i(request: Request):
    actor = actor_from_request(request)
    return {**actor.__dict__, "permissions": permissions_for_role(actor.role)}


@router.get("/workspaces")
async def list_workspaces(request: Request):
    require_permission(request, "workspace.read")
    return [item.model_dump(mode="json") for item in _store.list_workspaces()]


@router.post("/workspaces")
async def create_workspace(payload: WorkspaceRequest, request: Request):
    require_permission(request, "workspace.write")
    return _store.ensure_workspace(payload.workspace_id, payload.name).model_dump(mode="json")


@router.get("/workspaces/{workspace_id}")
async def get_workspace(workspace_id: str, request: Request):
    require_permission(request, "workspace.read")
    try:
        return _store.get_workspace(workspace_id).model_dump(mode="json")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/workspaces/{workspace_id}/fitness")
async def workspace_fitness(workspace_id: str, request: Request):
    require_permission(request, "health.read")
    try:
        return FitnessPolicy().assess(_store.get_workspace(workspace_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/workspaces/{workspace_id}/datasets/current")
async def register_current_dataset(workspace_id: str, request: Request, state: AppState = Depends(get_app_state)):
    require_permission(request, "dataset.write")
    if not state.has_data():
        raise HTTPException(status_code=400, detail="No dataset is currently loaded")
    csv_bytes = state.raw_df.to_csv(index=False).encode("utf-8")
    dataset = _store.register_dataset(
        workspace_id=workspace_id,
        source_name="current-loaded-dataset",
        content_hash=_store.hash_bytes(csv_bytes),
        row_count=len(state.raw_df),
        columns=list(state.raw_df.columns),
        quality={
            "missing_cells": int(state.raw_df.isna().sum().sum()),
            "duplicate_rows": int(state.raw_df.duplicated().sum()),
        },
    )
    return dataset.model_dump(mode="json")


@router.post("/workspaces/{workspace_id}/datasets/{dataset_id}/activate")
async def activate_dataset(workspace_id: str, dataset_id: str, request: Request):
    require_permission(request, "dataset.activate")
    try:
        return _store.activate_dataset(workspace_id, dataset_id).model_dump(mode="json")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/workspaces/{workspace_id}/models/train")
async def train_model(workspace_id: str, request: Request, state: AppState = Depends(get_app_state)):
    actor = require_permission(request, "model.train")
    workspace = _store.get_workspace(workspace_id)
    if not workspace.active_dataset_id:
        raise HTTPException(status_code=409, detail="Activate a dataset before training")
    if state.features_df is None or state.target_series is None:
        raise HTTPException(status_code=400, detail="Current dataset does not support predictive model training")

    idempotency_key = request.headers.get("idempotency-key") or f"train:{workspace_id}:{workspace.active_dataset_id}:{actor.actor_id}"
    job = _jobs.create(workspace_id=workspace_id, kind="model-training", idempotency_key=idempotency_key)
    if job.state == JobState.SUCCEEDED:
        return {"job": job.model_dump(mode="json"), "model_id": job.resource_id, "reused": True}
    _jobs.transition(job.job_id, JobState.RUNNING)
    try:
        model = ModelLifecycleService(_store).train(
            workspace_id, workspace.active_dataset_id, state.features_df, state.target_series
        )
        completed = _jobs.transition(job.job_id, JobState.SUCCEEDED, resource_id=model.model_id)
        return {"job": completed.model_dump(mode="json"), "model": model.model_dump(mode="json"), "reused": False}
    except Exception as exc:
        _jobs.transition(job.job_id, JobState.FAILED, error=str(exc))
        raise HTTPException(status_code=422, detail=f"Model training failed: {exc}") from exc


@router.post("/workspaces/{workspace_id}/models/{model_id}/activate")
async def activate_model(workspace_id: str, model_id: str, request: Request):
    require_permission(request, "model.activate")
    try:
        return ModelLifecycleService(_store).activate(workspace_id, model_id).model_dump(mode="json")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/sessions")
async def open_session(payload: SessionRequest, request: Request):
    require_permission(request, "session.write")
    try:
        return _store.open_session(
            workspace_id=payload.workspace_id, dataset_id=payload.dataset_id, model_id=payload.model_id
        ).model_dump(mode="json")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/jobs")
async def list_jobs(request: Request, workspace_id: Optional[str] = None):
    require_permission(request, "health.read")
    return [job.model_dump(mode="json") for job in _jobs.list(workspace_id)]


@router.get("/health")
async def platform_health(request: Request):
    require_permission(request, "health.read")
    health = SystemHealthMonitor(_store).check()
    health["interrupted_jobs"] = [job.job_id for job in _jobs.list() if job.state == JobState.RUNNING]
    return health


@router.post("/health/recover")
async def bounded_recovery(request: Request):
    require_permission(request, "health.recover")
    interrupted = _jobs.recover_interrupted()
    result = SystemHealthMonitor(_store).recover()
    result["failed_interrupted_jobs"] = interrupted
    return result
