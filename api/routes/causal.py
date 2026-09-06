"""Causal inference routes.

Disabled in the governed enterprise product until an explicit identification
strategy is provided (treatment definition, confounder set, overlap/positivity,
model diagnostics and sensitivity analysis). Observational HR data alone does not
justify causal intervention claims.
"""

from fastapi import APIRouter, Depends, HTTPException, Query

from api.dependencies import AppState, get_app_state

router = APIRouter(prefix="/api/causal", tags=["causal"])


def require_data(state: AppState = Depends(get_app_state)) -> AppState:
    if not state.has_data():
        if not state.load_from_database():
            raise HTTPException(status_code=400, detail="No data loaded. Please upload a file first.")
    return state


def _causal_unavailable() -> HTTPException:
    return HTTPException(
        status_code=409,
        detail=(
            "Causal intervention estimates are disabled until a validated identification design, confounder specification, overlap diagnostics, and sensitivity analysis are configured. Use observational association analysis instead."
        ),
    )


@router.get("/impact", deprecated=True)
async def get_causal_impact(
    treatment: str = Query(..., description="Treatment variable"),
    outcome: str = Query(default="Attrition", description="Outcome variable"),
    state: AppState = Depends(require_data),
):
    raise _causal_unavailable()


@router.get("/recommendations", deprecated=True)
async def get_intervention_recommendations(state: AppState = Depends(require_data)):
    raise _causal_unavailable()
