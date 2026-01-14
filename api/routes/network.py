"""Organizational Network Analysis API routes."""

from fastapi import APIRouter, HTTPException, Depends, Query

from src.database import Database
from src.network_engine import NetworkEngine
from api.dependencies import get_app_state, AppState

router = APIRouter(prefix="/api/network", tags=["network"])


def require_data(state: AppState = Depends(get_app_state)) -> AppState:
    """Dependency that requires data to be loaded."""
    if not state.has_data():
        if not state.load_from_database():
            raise HTTPException(
                status_code=400,
                detail="No data loaded. Please upload a file first."
            )
    return state


@router.get("/influencers")
async def get_key_influencers(
    limit: int = Query(default=10, ge=1, le=50, description="Number of influencers to return"),
    state: AppState = Depends(require_data)
):
    """
    Get key organizational influencers based on network centrality.
    
    Identifies employees whose departure would most impact the organization.
    """
    try:
        db = Database()
        df = db.get_df()
        
        engine = NetworkEngine(df)
        influencers = engine.get_key_influencers(limit=limit)
        
        return {
            'success': True,
            'influencers': influencers,
            'count': len(influencers)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/isolated")
async def get_isolated_employees(
    limit: int = Query(default=10, ge=1, le=50, description="Number of isolated employees to return"),
    state: AppState = Depends(require_data)
):
    """
    Get employees with low network connectivity.
    
    These employees may be at risk of disengagement.
    """
    try:
        db = Database()
        df = db.get_df()
        
        engine = NetworkEngine(df)
        isolated = engine.get_isolated_employees(limit=limit)
        
        return {
            'success': True,
            'isolated_employees': isolated,
            'count': len(isolated)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/summary")
async def get_network_summary(
    state: AppState = Depends(require_data)
):
    """
    Get organizational network health summary.
    
    Provides metrics on collaboration patterns and network density.
    """
    try:
        db = Database()
        df = db.get_df()
        
        engine = NetworkEngine(df)
        summary = engine.get_network_summary()
        
        return summary
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
