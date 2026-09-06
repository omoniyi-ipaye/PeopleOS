"""Workspace, dataset/model lifecycle, session and health control-plane API."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from api.dependencies import AppState, get_app_state
from src.platform.health import SystemHealthMonitor
from src.platform.model_lifecycle import ModelLifecycleService
from src.platform.workspace import WorkspaceStore

router = APIRouter(prefix="/api/platform", tags=["platform"])
_store = WorkspaceStore()


class WorkspaceRequest(BaseModel):
    workspace_id: str = Field(min_length=1, max_length=80, pattern=r"^[A-Za-z0-9_-]+$")
    name: str = Field(min_length=1, max_length=120)


class SessionRequest(BaseModel):
    workspace_id: str = "local"
    dataset_id: Optional[str] = None
    model_id: Optional[str] = None


@router.get("/workspaces")
async def list_workspaces():
    return [item.model_dump(mode="json") for item in _store.list_workspaces()]


@router.post("/workspaces")
async def create_workspace(request: WorkspaceRequest):
    return _store.ensure_workspace(request.workspace_id, request.name).model_dump(mode="json")


@router.get("/workspaces/{workspace_id}")
async def get_workspace(workspace_id: str):
    try:
        return _store.get_workspace(workspace_id).model_dump(mode="json")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/workspaces/{workspace_id}/datasets/current")
async def register_current_dataset(
    workspace_id: str,
    state: AppState = Depends(get_app_state),
):
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
async def activate_dataset(workspace_id: str, dataset_id: str):
    try:
        return _store.activate_dataset(workspace_id, dataset_id).model_dump(mode="json")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/workspaces/{workspace_id}/models/train")
async def train_model(
    workspace_id: str,
    state: AppState = Depends(get_app_state),
):
    workspace = _store.get_workspace(workspace_id)
    if not workspace.active_dataset_id:
        raise HTTPException(status_code=409, detail="Activate a dataset before training")
    if state.features_df is None or state.target_series is None:
        raise HTTPException(status_code=400, detail="Current dataset does not support predictive model training")
    lifecycle = ModelLifecycleService(_store)
    try:
        model = lifecycle.train(
            workspace_id,
            workspace.active_dataset_id,
            state.features_df,
            state.target_series,
        )
        return model.model_dump(mode="json")
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Model training failed: {exc}") from exc


@router.post("/workspaces/{workspace_id}/models/{model_id}/activate")
async def activate_model(workspace_id: str, model_id: str):
    try:
        return ModelLifecycleService(_store).activate(workspace_id, model_id).model_dump(mode="json")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/sessions")
async def open_session(request: SessionRequest):
    try:
        session = _store.open_session(
            workspace_id=request.workspace_id,
            dataset_id=request.dataset_id,
            model_id=request.model_id,
        )
        return session.model_dump(mode="json")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/health")
async def platform_health():
    return SystemHealthMonitor(_store).check()


@router.post("/health/recover")
async def bounded_recovery():
    return SystemHealthMonitor(_store).recover()
