"""Causal Inference API routes."""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Optional, List

from src.database import Database
from src.causal_engine import CausalEngine
from api.dependencies import get_app_state, AppState

router = APIRouter(prefix="/api/causal", tags=["causal"])


def require_data(state: AppState = Depends(get_app_state)) -> AppState:
    """Dependency that requires data to be loaded."""
    if not state.has_data():
        if not state.load_from_database():
            raise HTTPException(
                status_code=400,
                detail="No data loaded. Please upload a file first."
            )
    return state


@router.get("/impact")
async def get_causal_impact(
    treatment: str = Query(..., description="Treatment variable (e.g., 'HighSalary', 'Tenure')"),
    outcome: str = Query(default="Attrition", description="Outcome variable"),
    state: AppState = Depends(require_data)
):
    """
    Estimate the causal impact of an intervention on an outcome.
    
    Returns HR-friendly interpretation with confidence levels.
    
    Example: `/api/causal/impact?treatment=HighSalary&outcome=Attrition`
    """
    try:
        db = Database()
        df = db.get_df()
        
        engine = CausalEngine(df)
        result = engine.estimate_intervention_effect(treatment, outcome)
        
        if not result.get('success'):
            raise HTTPException(status_code=400, detail=result.get('reason', 'Estimation failed'))
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recommendations")
async def get_intervention_recommendations(
    state: AppState = Depends(require_data)
):
    """
    Get ranked list of intervention recommendations based on causal analysis.
    
    Tests common HR interventions and ranks by estimated impact on retention.
    """
    try:
        db = Database()
        df = db.get_df()
        
        engine = CausalEngine(df)
        recommendations = engine.get_intervention_recommendations()
        
        return {
            'success': True,
            'recommendations': recommendations,
            'count': len(recommendations)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
