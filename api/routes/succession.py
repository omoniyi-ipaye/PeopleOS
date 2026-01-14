"""Succession planning route handlers."""

from typing import List, Dict, Any
import pandas as pd

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from api.dependencies import get_app_state, AppState

router = APIRouter(prefix="/api/succession", tags=["succession"])


class ReadinessScore(BaseModel):
    """Employee readiness score."""
    employee_id: str
    dept: str
    tenure: float
    last_rating: float
    readiness_score: float
    readiness_level: str


class HighPotential(BaseModel):
    """High-potential employee."""
    employee_id: str
    dept: str
    tenure: float
    last_rating: float
    potential_level: str
    attrition_risk: str


class BenchStrength(BaseModel):
    """Department bench strength."""
    dept: str
    bench_strength: float
    ready_now: int
    ready_soon: int
    developing: int
    total: int
    status: str


class NineBoxEntry(BaseModel):
    """Nine-box matrix entry."""
    employee_id: str
    dept: str
    last_rating: float
    performance: str
    potential: str
    nine_box: str


class NineBoxSummary(BaseModel):
    """Nine-box category summary."""
    category: str
    count: int
    percentage: float


def require_succession(state: AppState = Depends(get_app_state)) -> AppState:
    """Dependency that requires succession engine."""
    if not state.has_data():
        if not state.load_from_database():
            raise HTTPException(
                status_code=400,
                detail="No data loaded. Please upload a file first."
            )

    if state.succession_engine is None:
        raise HTTPException(
            status_code=500,
            detail="Succession engine not initialized"
        )

    return state


@router.get("/readiness", response_model=List[ReadinessScore])
async def get_readiness_scores(
    state: AppState = Depends(require_succession)
) -> List[ReadinessScore]:
    """
    Get readiness scores for all employees.
    """
    readiness_df = state.succession_engine.calculate_readiness_scores()

    scores = []
    for _, row in readiness_df.iterrows():
        scores.append(ReadinessScore(
            employee_id=row['EmployeeID'],
            dept=row['Dept'],
            tenure=float(row['Tenure']),
            last_rating=float(row['LastRating']),
            readiness_score=float(row['ReadinessScore']),
            readiness_level=row['ReadinessLevel']
        ))

    return scores


@router.get("/high-potentials", response_model=List[HighPotential])
async def get_high_potentials(
    state: AppState = Depends(require_succession)
) -> List[HighPotential]:
    """
    Get high-potential employees identified for succession.
    """
    hp_df = state.succession_engine.identify_high_potentials()

    potentials = []
    for _, row in hp_df.iterrows():
        potentials.append(HighPotential(
            employee_id=row['EmployeeID'],
            dept=row['Dept'],
            tenure=float(row['Tenure']),
            last_rating=float(row['LastRating']),
            potential_level=row['PotentialLevel'],
            attrition_risk=row.get('AttritionRisk', 'N/A')
        ))

    return potentials


@router.get("/pipeline")
async def get_succession_pipeline(
    state: AppState = Depends(require_succession)
) -> Dict[str, Any]:
    """
    Get succession pipeline by department.
    """
    return state.succession_engine.get_succession_pipeline()


@router.get("/bench-strength", response_model=List[BenchStrength])
async def get_bench_strength(
    state: AppState = Depends(require_succession)
) -> List[BenchStrength]:
    """
    Get bench strength scores by department.
    """
    bench_df = state.succession_engine.calculate_bench_strength()

    strengths = []
    for _, row in bench_df.iterrows():
        strengths.append(BenchStrength(
            dept=row['Dept'],
            bench_strength=float(row['BenchStrength']),
            ready_now=int(row['ReadyNow']),
            ready_soon=int(row['ReadySoon']),
            developing=int(row['Developing']),
            total=int(row['Total']),
            status=row['Status']
        ))

    return strengths


@router.get("/gaps")
async def get_critical_gaps(
    state: AppState = Depends(require_succession)
) -> List[Dict[str, Any]]:
    """
    Get departments with succession gaps.
    """
    gaps_df = state.succession_engine.identify_critical_gaps()

    gaps = []
    for _, row in gaps_df.iterrows():
        gaps.append({
            'dept': row['Dept'],
            'bench_strength': float(row['BenchStrength']),
            'ready_now': int(row['ReadyNow']),
            'ready_soon': int(row['ReadySoon']),
            'gap_severity': row['GapSeverity'],
            'recommendation': row['Recommendation']
        })

    return gaps


@router.get("/recommendations")
async def get_retention_recommendations(
    state: AppState = Depends(require_succession)
) -> List[Dict[str, Any]]:
    """
    Get retention recommendations for high-potential employees.
    """
    return state.succession_engine.get_retention_recommendations()


@router.get("/9box", response_model=List[NineBoxEntry])
async def get_9box_matrix(
    state: AppState = Depends(require_succession)
) -> List[NineBoxEntry]:
    """
    Get 9-box matrix classification for all employees.
    """
    ninebox_df = state.succession_engine.get_9box_matrix()

    entries = []
    for _, row in ninebox_df.iterrows():
        entries.append(NineBoxEntry(
            employee_id=row['EmployeeID'],
            dept=row['Dept'],
            last_rating=float(row['LastRating']),
            performance=str(row['Performance']),
            potential=str(row['Potential']),
            nine_box=row['NineBox']
        ))

    return entries


@router.get("/9box/summary", response_model=List[NineBoxSummary])
async def get_9box_summary(
    state: AppState = Depends(require_succession)
) -> List[NineBoxSummary]:
    """
    Get 9-box matrix summary counts.
    """
    summary_df = state.succession_engine.get_9box_summary()

    summaries = []
    for _, row in summary_df.iterrows():
        summaries.append(NineBoxSummary(
            category=row['Category'],
            count=int(row['Count']),
            percentage=float(row['Percentage'])
        ))

    return summaries


@router.get("/summary")
async def get_succession_summary(
    state: AppState = Depends(require_succession)
) -> Dict[str, Any]:
    """
    Get overall succession planning summary.
    """
    # Analyze all succession metrics
    analysis = state.succession_engine.analyze_all()
    
    # helper to safely serialize potential DataFrames
    def safe_serialize(obj):
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        if isinstance(obj, (list, dict, int, float, str, bool, type(None))):
            return obj
        return str(obj)

    # Extract key summary metrics
    summary = {
        "total_employees": int(len(state.succession_engine.df)),
        "high_potentials_count": len(analysis.get("high_potentials", [])),
        "ready_now_count": sum(1 for s in analysis.get("readiness_scores", []).to_dict('records') if s.get("ReadinessLevel") == "Ready Now") if isinstance(analysis.get("readiness_scores"), pd.DataFrame) else 0,
        "critical_gaps": len(analysis.get("critical_gaps", [])),
        "nine_box_summary": safe_serialize(analysis.get("nine_box_summary")),
        "bench_strength": safe_serialize(analysis.get("bench_strength")),
    }
    
    return summary
