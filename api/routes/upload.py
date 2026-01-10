"""Upload route handlers."""

import os
import tempfile
from typing import Dict, Any

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel

from api.dependencies import get_app_state, AppState

router = APIRouter(prefix="/api/upload", tags=["upload"])


@router.get("/template")
async def download_template():
    """
    Download a CSV template for the Golden Schema.
    """
    template_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "data", "templates", "peopleos_template.csv"
    )

    if not os.path.exists(template_path):
        raise HTTPException(status_code=404, detail="Template file not found")

    return FileResponse(
        template_path,
        media_type="text/csv",
        filename="peopleos_golden_schema_template.csv"
    )


class UploadResponse(BaseModel):
    """Response for data upload."""
    model_config = {'protected_namespaces': ()}
    success: bool
    message: str
    rows_loaded: int
    columns: list[str]
    features_enabled: Dict[str, bool]


class DatabaseStatusResponse(BaseModel):
    """Response for database status check."""
    has_data: bool
    employee_count: int
    features_enabled: Dict[str, bool]


@router.post("", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    state: AppState = Depends(get_app_state)
) -> UploadResponse:
    """
    Upload a CSV or JSON file with employee data.

    The file will be validated against the Golden Schema and merged
    with any existing data in the database.
    """
    # Validate file type
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    ext = file.filename.split(".")[-1].lower()
    if ext not in ["csv", "json"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Use CSV or JSON."
        )

    # Save to temp file
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Load data
        result = state.load_data(tmp_path, file.filename)

        return UploadResponse(
            success=True,
            message=f"Successfully loaded {result['rows_loaded']} employees",
            rows_loaded=result['rows_loaded'],
            columns=result['columns'],
            features_enabled=result['features_enabled']
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    finally:
        # Clean up temp file
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@router.get("/status", response_model=DatabaseStatusResponse)
async def get_database_status(
    state: AppState = Depends(get_app_state)
) -> DatabaseStatusResponse:
    """
    Check if data is loaded and get current status.
    """
    if not state.has_data():
        # Try to load from database
        loaded = state.load_from_database()
        if not loaded:
            return DatabaseStatusResponse(
                has_data=False,
                employee_count=0,
                features_enabled=state.features_enabled
            )

    return DatabaseStatusResponse(
        has_data=True,
        employee_count=len(state.raw_df) if state.raw_df is not None else 0,
        features_enabled=state.features_enabled
    )


@router.post("/load-sample")
async def load_sample_data(
    state: AppState = Depends(get_app_state)
) -> UploadResponse:
    """
    Load the sample HR data file for demo purposes.
    """
    sample_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "sample_hr_data.csv"
    )

    if not os.path.exists(sample_path):
        raise HTTPException(status_code=404, detail="Sample data file not found")

    try:
        result = state.load_data(sample_path, "sample_hr_data.csv")

        return UploadResponse(
            success=True,
            message=f"Successfully loaded {result['rows_loaded']} employees from sample data",
            rows_loaded=result['rows_loaded'],
            columns=result['columns'],
            features_enabled=result['features_enabled']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reset")
async def reset_data(
    state: AppState = Depends(get_app_state)
) -> Dict[str, Any]:
    """
    Reset all loaded data and engines.
    """
    state.reset()
    return {"success": True, "message": "All data has been reset"}
