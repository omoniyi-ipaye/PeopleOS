"""
FastAPI application entry point for PeopleOS.

Run locally with: uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes.upload import router as upload_router
from api.routes.analytics import router as analytics_router
from api.routes.predictions import router as predictions_router
from api.routes.compensation import router as compensation_router
from api.routes.succession import router as succession_router
from api.routes.team import router as team_router
from api.routes.fairness import router as fairness_router
from api.routes.search import router as search_router
from api.routes.advisor import router as advisor_router
from api.routes.sessions import router as sessions_router
from api.routes.nlp import router as nlp_router
from api.routes.survival import router as survival_router
from api.routes.quality_of_hire import router as quality_of_hire_router
from api.routes.structural import router as structural_router
from api.routes.sentiment import router as sentiment_router
from api.routes.experience import router as experience_router
from api.routes.scenario import router as scenario_router
from api.routes.model_lab import router as model_lab_router
from api.routes.geo import router as geo_router
from api.routes.causal import router as causal_router
from api.routes.network import router as network_router
from api.routes.intelligence import router as intelligence_router
from api.routes.platform import router as platform_router
from api.routes.desktop import router as desktop_router
from api.dependencies import get_app_state
from api.runtime_registry import get_local_state, get_workspace_state
from api.security import local_first_access_guard
from src.logger import get_logger
from src.platform.health import SystemHealthMonitor
from src.platform.workspace import WorkspaceStore
from src.serialization import json_safe
from src.platform.provenance import runtime_integrity

logger = get_logger('api_main')

app = FastAPI(
    title="PeopleOS API",
    description="""
    PeopleOS API - HR Analytics and Governed People Intelligence Backend

    A comprehensive REST API for workforce analytics providing:
    - **Analytics**: current population, observed attrition and department statistics
    - **Predictions**: governed aggregate predictive retention signals after explicit model activation
    - **Compensation**: current compensation summaries and disparity screening
    - **Team Dynamics**: aggregate organizational patterns
    - **Fairness**: screening metrics with minimum-group controls
    - **Semantic Search**: evidence retrieval when supported text is indexed
    - **People Intelligence Agent**: governed evidence-based workforce investigation
    - **Workspace control plane**: explicit dataset/model/session lifecycle
    - **System health**: deterministic fitness checks and bounded recovery
    - **Survival Analysis**: cohort-level Kaplan-Meier and Cox association analysis
    - **Quality of Hire**: pre-hire to post-hire association analysis

    Consequential individual ranking and unsupported causal intervention claims are
    intentionally excluded from the enterprise product boundary.
    """,
    version="3.0.0-transition",
    docs_url="/docs",
    redoc_url="/redoc"
)

from api.integrity import evidence_snapshot_guard
app.middleware("http")(evidence_snapshot_guard)
app.middleware("http")(local_first_access_guard)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.dependency_overrides[get_app_state] = get_workspace_state

app.include_router(upload_router)
app.include_router(analytics_router)
app.include_router(predictions_router)
app.include_router(compensation_router)
app.include_router(succession_router)
app.include_router(team_router)
app.include_router(fairness_router)
app.include_router(search_router)
app.include_router(advisor_router)
app.include_router(sessions_router)
app.include_router(nlp_router)
app.include_router(survival_router)
app.include_router(quality_of_hire_router)
app.include_router(structural_router)
app.include_router(sentiment_router)
app.include_router(experience_router)
app.include_router(scenario_router)
app.include_router(model_lab_router)
app.include_router(geo_router)
app.include_router(causal_router)
app.include_router(network_router)
app.include_router(intelligence_router)
app.include_router(platform_router)
app.include_router(desktop_router)


@app.get("/")
async def root():
    return {
        "name": "PeopleOS API",
        "version": "3.0.0-transition",
        "docs": "/docs",
        "status": "running"
    }


@app.get("/api/health")
async def health_check():
    state = get_local_state()
    platform_health = SystemHealthMonitor(WorkspaceStore()).check()
    payload = {
        "status": "healthy" if platform_health["status"] == "healthy" else "degraded",
        "data_loaded": state.has_data(),
        "features_enabled": state.features_enabled,
        "platform": platform_health,
    }
    return JSONResponse(content=json_safe(payload))


@app.get("/api/status")
async def api_status():
    """Return operational capability state without exposing dataset schema or model metrics."""
    state = get_local_state()
    workspace = WorkspaceStore().get_workspace("local")

    integrity = runtime_integrity(state, workspace)
    payload = {
        "integrity": integrity,
        "status": "running",
        "data": {
            "loaded": state.has_data(),
            "row_count": len(state.raw_df) if state.raw_df is not None else 0,
            "active_dataset": workspace.active_dataset_id is not None,
        },
        "capabilities": {
            "analytics": state.analytics_engine is not None,
            "predictive_model": integrity["model_ready"],
            "compensation": state.compensation_engine is not None,
            "fairness": state.fairness_engine is not None,
            "vector_search": bool(state.vector_engine is not None and state.vector_engine.is_initialized()),
            "llm": bool(state.llm_client is not None and state.features_enabled.get('llm', False)),
            "people_intelligence": True,
            "workspace_control_plane": True,
        },
        "workspace": {
            "active_dataset": workspace.active_dataset_id is not None,
            "active_model": workspace.active_model_id is not None,
            "dataset_versions": len(workspace.datasets),
            "model_versions": len(workspace.models),
            "sessions": len(workspace.sessions),
        },
    }
    return JSONResponse(content=json_safe(payload))


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.exception(
        "Unhandled API error on %s %s",
        getattr(request, 'method', 'UNKNOWN'),
        getattr(request, 'url', 'UNKNOWN'),
        exc_info=exc,
    )
    return JSONResponse(status_code=500, content={"error": "Internal server error"})


if __name__ == "__main__":
    import uvicorn

    api_host = os.getenv("PEOPLEOS_API_HOST", "127.0.0.1")
    api_port = int(os.getenv("PEOPLEOS_API_PORT", "8000"))
    uvicorn.run(app, host=api_host, port=api_port, reload=True)
