"""
FastAPI application entry point for PeopleOS.

Run locally with: uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
"""

import os
import sys

# Add parent directory to path for src imports
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
from api.dependencies import get_app_state
from api.security import local_first_access_guard


app = FastAPI(
    title="PeopleOS API",
    description="""
    PeopleOS API - HR Analytics and Governed People Intelligence Backend

    A comprehensive REST API for HR analytics providing:
    - **Analytics**: Headcount, turnover, department statistics
    - **Predictions**: ML-based attrition risk with SHAP explanations
    - **Compensation**: Pay equity analysis, salary benchmarking
    - **Succession**: Readiness scoring, 9-box analysis
    - **Team Dynamics**: Team health, diversity metrics
    - **Fairness**: EEOC compliance, four-fifths rule
    - **Semantic Search**: Performance review search
    - **AI Advisor**: LLM-powered strategic insights
    - **People Intelligence Agent**: governed evidence-based workforce investigation
    - **Survival Analysis**: Kaplan-Meier curves, Cox PH flight risk modeling
    - **Quality of Hire**: Pre-hire to post-hire correlation analysis
    - **Causal Inference**: Intervention impact estimation (What-If analysis)
    """,
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Any non-loopback request is denied unless PEOPLEOS_API_TOKEN is explicitly
# configured and supplied as a Bearer token. This protects local-first data even
# if the process is accidentally bound to a wider network interface.
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


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "PeopleOS API",
        "version": "2.1.0",
        "docs": "/docs",
        "status": "running"
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    state = get_app_state()
    return {
        "status": "healthy",
        "data_loaded": state.has_data(),
        "features_enabled": state.features_enabled
    }


@app.get("/api/status")
async def api_status():
    """Get comprehensive API status."""
    state = get_app_state()

    status = {
        "data": {
            "loaded": state.has_data(),
            "row_count": len(state.raw_df) if state.raw_df is not None else 0,
            "columns": list(state.raw_df.columns) if state.raw_df is not None else []
        },
        "engines": {
            "analytics": state.analytics_engine is not None,
            "ml": state.ml_engine is not None and state.ml_engine.is_trained,
            "compensation": state.compensation_engine is not None,
            "succession": state.succession_engine is not None,
            "team_dynamics": state.team_dynamics_engine is not None,
            "fairness": state.fairness_engine is not None,
            "vector_search": state.vector_engine is not None and state.vector_engine.is_initialized(),
            "llm": state.llm_client is not None and state.features_enabled.get('llm', False),
            "survival": state.survival_engine is not None,
            "quality_of_hire": state.quality_of_hire_engine is not None,
            "structural": state.structural_engine is not None,
            "sentiment": state.sentiment_engine is not None,
            "experience": state.experience_engine is not None,
            "scenario": state.scenario_engine is not None,
            "people_intelligence": True,
        },
        "features_enabled": state.features_enabled,
        "model_metrics": state.model_metrics
    }

    return status


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc)
        }
    )


if __name__ == "__main__":
    import uvicorn

    # Local-first default. Set PEOPLEOS_API_HOST explicitly (for example,
    # 0.0.0.0) only when intentionally exposing the API beyond this machine.
    api_host = os.getenv("PEOPLEOS_API_HOST", "127.0.0.1")
    api_port = int(os.getenv("PEOPLEOS_API_PORT", "8000"))
    uvicorn.run(app, host=api_host, port=api_port, reload=True)
