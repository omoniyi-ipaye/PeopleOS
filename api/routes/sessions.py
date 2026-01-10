"""
Session management routes for PeopleOS API.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from src.session_manager import get_session_manager
from api.dependencies import get_app_state

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


class SaveSessionRequest(BaseModel):
    """Request model for saving a session."""
    name: str


class SessionSummary(BaseModel):
    """Summary of a saved session."""
    session_id: str
    session_name: str
    created_at: str
    row_count: int
    filepath: str
    features_enabled: Optional[dict] = None
    analytics_summary: Optional[dict] = None


@router.get("")
async def list_sessions():
    """List all saved sessions."""
    manager = get_session_manager()
    sessions = manager.list_sessions()

    return {
        "sessions": sessions,
        "count": len(sessions),
    }


@router.post("")
async def save_session(request: SaveSessionRequest):
    """Save the current session."""
    state = get_app_state()

    if state.df is None:
        raise HTTPException(status_code=400, detail="No data loaded to save")

    manager = get_session_manager()

    # Gather analytics data
    analytics_data = {}
    if state.analytics_engine:
        summary = state.analytics_engine.calculate_summary()
        analytics_data = {
            "headcount": summary.get("headcount"),
            "department_count": summary.get("department_count"),
            "turnover_rate": summary.get("turnover_rate"),
            "tenure_mean": summary.get("avg_tenure"),
        }

    # Gather ML data
    ml_data = {}
    if state.ml_engine:
        metrics = state.ml_engine.get_model_metrics()
        if metrics:
            ml_data["metrics"] = metrics
            ml_data["model_trained"] = True

    filepath = manager.save_session(
        session_name=request.name,
        raw_data=state.df,
        analytics_data=analytics_data,
        ml_data=ml_data,
        features_enabled=state.features_enabled or {}
    )

    return {
        "success": True,
        "message": f"Session '{request.name}' saved successfully",
        "filepath": filepath,
    }


@router.post("/load")
async def load_session(filepath: str):
    """Load a saved session."""
    manager = get_session_manager()
    session_data = manager.load_session(filepath)

    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found")

    # Load the data into the application state
    state = get_app_state()
    df = session_data.get("raw_data")

    if df is not None:
        state.set_data(df)

        return {
            "success": True,
            "message": f"Session '{session_data.get('session_name')}' loaded successfully",
            "session": {
                "session_id": session_data.get("session_id"),
                "session_name": session_data.get("session_name"),
                "row_count": len(df),
                "features_enabled": session_data.get("features_enabled", {}),
            },
        }
    else:
        raise HTTPException(status_code=400, detail="Session has no data")


@router.delete("")
async def delete_session(filepath: str):
    """Delete a saved session."""
    manager = get_session_manager()
    success = manager.delete_session(filepath)

    if not success:
        raise HTTPException(status_code=404, detail="Session not found or could not be deleted")

    return {
        "success": True,
        "message": "Session deleted successfully",
    }
