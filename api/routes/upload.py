"""Upload route handlers."""

import os
import tempfile
from typing import Dict, Any, Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel

from api.dependencies import get_app_state, AppState
from src.platform.runtime_loader import load_dataset
from src.platform.workspace import WorkspaceStore

router = APIRouter(prefix="/api/upload", tags=["upload"])
_store = WorkspaceStore()


@router.get("/template")
async def download_template():
    template_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "data", "templates", "peopleos_template.csv"
    )
    if not os.path.exists(template_path):
        raise HTTPException(status_code=404, detail="Template file not found")
    return FileResponse(template_path, media_type="text/csv", filename="peopleos_golden_schema_template.csv")


class UploadResponse(BaseModel):
    model_config = {'protected_namespaces': ()}
    success: bool
    message: str
    rows_loaded: int
    columns: list[str]
    features_enabled: Dict[str, bool]
    workspace_id: str = "local"
    dataset_id: Optional[str] = None
    dataset_version: Optional[int] = None
    deferred: Dict[str, bool] = {}


class DatabaseStatusResponse(BaseModel):
    has_data: bool
    employee_count: int
    features_enabled: Dict[str, bool]
    workspace_id: str = "local"
    active_dataset_id: Optional[str] = None


def _register_loaded_dataset(state: AppState, source_name: str, content_hash: str, workspace_id: str = "local"):
    dataset = _store.register_dataset(
        workspace_id=workspace_id,
        source_name=source_name,
        content_hash=content_hash,
        row_count=len(state.raw_df) if state.raw_df is not None else 0,
        columns=list(state.raw_df.columns) if state.raw_df is not None else [],
        quality={
            "missing_cells": int(state.raw_df.isna().sum().sum()) if state.raw_df is not None else 0,
            "duplicate_rows": int(state.raw_df.duplicated().sum()) if state.raw_df is not None else 0,
        },
    )
    return _store.activate_dataset(workspace_id, dataset.dataset_id)


@router.post("", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...), state: AppState = Depends(get_app_state)) -> UploadResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    ext = file.filename.split(".")[-1].lower()
    if ext not in ["csv", "json"]:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}. Use CSV or JSON.")

    try:
        content = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        result = load_dataset(state, tmp_path, file.filename)
        dataset = _register_loaded_dataset(state, file.filename, _store.hash_bytes(content))
        return UploadResponse(
            success=True,
            message=f"Successfully activated {result['rows_loaded']} employees as dataset v{dataset.version}",
            rows_loaded=result['rows_loaded'],
            columns=result['columns'],
            features_enabled=result['features_enabled'],
            dataset_id=dataset.dataset_id,
            dataset_version=dataset.version,
            deferred=result.get('deferred', {}),
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@router.get("/status", response_model=DatabaseStatusResponse)
async def get_database_status(state: AppState = Depends(get_app_state)) -> DatabaseStatusResponse:
    if not state.has_data():
        loaded = state.load_from_database()
        if not loaded:
            workspace = _store.ensure_workspace("local", "Local workspace")
            return DatabaseStatusResponse(
                has_data=False,
                employee_count=0,
                features_enabled=state.features_enabled,
                active_dataset_id=workspace.active_dataset_id,
            )
    workspace = _store.ensure_workspace("local", "Local workspace")
    return DatabaseStatusResponse(
        has_data=True,
        employee_count=len(state.raw_df) if state.raw_df is not None else 0,
        features_enabled=state.features_enabled,
        active_dataset_id=workspace.active_dataset_id,
    )


@router.post("/load-sample", response_model=UploadResponse)
async def load_sample_data(state: AppState = Depends(get_app_state)) -> UploadResponse:
    sample_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "sample_hr_data.csv"
    )
    if not os.path.exists(sample_path):
        raise HTTPException(status_code=404, detail="Sample data file not found")
    try:
        content = open(sample_path, "rb").read()
        result = load_dataset(state, sample_path, "sample_hr_data.csv")
        dataset = _register_loaded_dataset(state, "sample_hr_data.csv", _store.hash_bytes(content))
        return UploadResponse(
            success=True,
            message=f"Successfully activated {result['rows_loaded']} employees from sample data as dataset v{dataset.version}",
            rows_loaded=result['rows_loaded'],
            columns=result['columns'],
            features_enabled=result['features_enabled'],
            dataset_id=dataset.dataset_id,
            dataset_version=dataset.version,
            deferred=result.get('deferred', {}),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reset")
async def reset_data(state: AppState = Depends(get_app_state)) -> Dict[str, Any]:
    state.reset()
    return {
        "success": True,
        "message": "Runtime data has been reset. Workspace lifecycle history is retained for auditability.",
    }
