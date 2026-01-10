"""Team dynamics route handlers."""

from typing import List, Dict, Any

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from api.dependencies import get_app_state, AppState

router = APIRouter(prefix="/api/team", tags=["team"])


class TeamHealth(BaseModel):
    """Team health score."""
    dept: str
    health_score: float
    avg_tenure: float | None
    avg_rating: float | None
    headcount: int
    attrition_rate: float | None
    status: str


class DiversityMetrics(BaseModel):
    """Diversity metrics for a department."""
    dept: str
    headcount: int
    tenure_diversity: float | None
    age_diversity: float | None
    salary_equity: float | None
    overall_diversity: float


def require_team(state: AppState = Depends(get_app_state)) -> AppState:
    """Dependency that requires team dynamics engine."""
    if not state.has_data():
        if not state.load_from_database():
            raise HTTPException(
                status_code=400,
                detail="No data loaded. Please upload a file first."
            )

    if state.team_dynamics_engine is None:
        raise HTTPException(
            status_code=500,
            detail="Team dynamics engine not initialized"
        )

    return state


@router.get("/health", response_model=List[TeamHealth])
async def get_team_health(
    state: AppState = Depends(require_team)
) -> List[TeamHealth]:
    """
    Get health scores for all teams/departments.
    """
    health_df = state.team_dynamics_engine.calculate_team_health_scores()

    teams = []
    for _, row in health_df.iterrows():
        teams.append(TeamHealth(
            dept=row['Dept'],
            health_score=float(row['HealthScore']),
            avg_tenure=float(row['AvgTenure']) if row.get('AvgTenure') is not None else None,
            avg_rating=float(row['AvgRating']) if row.get('AvgRating') is not None else None,
            headcount=int(row['Headcount']),
            attrition_rate=float(row['AttritionRate']) if row.get('AttritionRate') is not None else None,
            status=row['Status']
        ))

    return teams


@router.get("/diversity", response_model=List[DiversityMetrics])
async def get_diversity_metrics(
    state: AppState = Depends(require_team)
) -> List[DiversityMetrics]:
    """
    Get diversity metrics for all teams/departments.
    """
    diversity_df = state.team_dynamics_engine.analyze_team_diversity()

    metrics = []
    for _, row in diversity_df.iterrows():
        metrics.append(DiversityMetrics(
            dept=row['Dept'],
            headcount=int(row['Headcount']),
            tenure_diversity=float(row['TenureDiversity']) if 'TenureDiversity' in row else None,
            age_diversity=float(row['AgeDiversity']) if 'AgeDiversity' in row else None,
            salary_equity=float(row['SalaryEquity']) if 'SalaryEquity' in row else None,
            overall_diversity=float(row['OverallDiversity'])
        ))

    return metrics


@router.get("/analysis")
async def get_team_analysis(
    state: AppState = Depends(require_team)
) -> Dict[str, Any]:
    """
    Get full team dynamics analysis.
    """
    analysis = state.team_dynamics_engine.analyze_all()

    # Convert DataFrames to lists of dicts
    result = {
        'health': [],
        'diversity': [],
        'at_risk_teams': [],
        'summary': analysis.get('summary', {})
    }

    if 'health' in analysis and not analysis['health'].empty:
        for _, row in analysis['health'].iterrows():
            result['health'].append({
                'dept': row['Dept'],
                'health_score': float(row['HealthScore']),
                'avg_tenure': float(row['AvgTenure']) if row.get('AvgTenure') is not None else None,
                'avg_rating': float(row['AvgRating']) if row.get('AvgRating') is not None else None,
                'headcount': int(row['Headcount']),
                'attrition_rate': float(row['AttritionRate']) if row.get('AttritionRate') is not None else None,
                'status': row['Status']
            })

    if 'diversity' in analysis and not analysis['diversity'].empty:
        for _, row in analysis['diversity'].iterrows():
            result['diversity'].append({
                'dept': row['Dept'],
                'headcount': int(row['Headcount']),
                'tenure_diversity': float(row['TenureDiversity']) if 'TenureDiversity' in row else None,
                'age_diversity': float(row['AgeDiversity']) if 'AgeDiversity' in row else None,
                'salary_equity': float(row['SalaryEquity']) if 'SalaryEquity' in row else None,
                'overall_diversity': float(row['OverallDiversity'])
            })

    if 'at_risk' in analysis and not analysis['at_risk'].empty:
        for _, row in analysis['at_risk'].iterrows():
            result['at_risk_teams'].append({
                'dept': row['Dept'],
                'headcount': int(row['Headcount']),
                'health_score': float(row['HealthScore']),
                'status': row['Status'],
                'risk_factors': row.get('RiskFactors', ''),
                'recommendations': row.get('Recommendations', '')
            })

    return result
