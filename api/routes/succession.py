"""Governed aggregate Succession Planning routes.

PeopleOS may surface department-level bench and gap summaries for planning, but it
does not rank, label, or recommend individual employees as high potential,
promotion-ready, successors, or retention targets.
"""

from typing import Any, Dict, List

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.dependencies import AppState, get_app_state

router = APIRouter(prefix="/api/succession", tags=["succession"])


class BenchStrength(BaseModel):
    dept: str
    bench_strength: float
    ready_now: int
    ready_soon: int
    developing: int
    total: int
    status: str


class NineBoxSummary(BaseModel):
    category: str
    count: int
    percentage: float


def require_succession(state: AppState = Depends(get_app_state)) -> AppState:
    if not state.has_data():
        if not state.load_from_database():
            raise HTTPException(status_code=400, detail="No data loaded. Please upload a file first.")
    if state.succession_engine is None:
        raise HTTPException(status_code=400, detail="Succession analysis is unavailable for the current dataset.")
    return state


def _individual_disabled() -> HTTPException:
    return HTTPException(
        status_code=403,
        detail="Individual succession ranking is disabled. Use aggregate department bench and gap analysis.",
    )


@router.get("/readiness", deprecated=True)
async def get_readiness_scores(state: AppState = Depends(require_succession)):
    raise _individual_disabled()


@router.get("/high-potentials", deprecated=True)
async def get_high_potentials(state: AppState = Depends(require_succession)):
    raise _individual_disabled()


@router.get("/pipeline", deprecated=True)
async def get_succession_pipeline(state: AppState = Depends(require_succession)):
    raise _individual_disabled()


@router.get("/bench-strength", response_model=List[BenchStrength])
async def get_bench_strength(state: AppState = Depends(require_succession)) -> List[BenchStrength]:
    frame = state.succession_engine.calculate_bench_strength()
    return [
        BenchStrength(
            dept=row['Dept'],
            bench_strength=float(row['BenchStrength']),
            ready_now=int(row['ReadyNow']),
            ready_soon=int(row['ReadySoon']),
            developing=int(row['Developing']),
            total=int(row['Total']),
            status=row['Status'],
        )
        for _, row in frame.iterrows()
    ]


@router.get("/gaps")
async def get_critical_gaps(state: AppState = Depends(require_succession)) -> List[Dict[str, Any]]:
    frame = state.succession_engine.identify_critical_gaps()
    output = []
    for _, row in frame.iterrows():
        output.append({
            'dept': row['Dept'],
            'bench_strength': float(row['BenchStrength']),
            'ready_now': int(row['ReadyNow']),
            'ready_soon': int(row['ReadySoon']),
            'gap_severity': row['GapSeverity'],
            'recommendation': 'Review role coverage and succession process with accountable People leadership.',
            'metric_semantics': 'aggregate_heuristic_not_individual_readiness_determination',
        })
    return output


@router.get("/recommendations", deprecated=True)
async def get_retention_recommendations(state: AppState = Depends(require_succession)):
    raise _individual_disabled()


@router.get("/9box", deprecated=True)
async def get_9box_matrix(state: AppState = Depends(require_succession)):
    raise _individual_disabled()


@router.get("/9box/summary", response_model=List[NineBoxSummary])
async def get_9box_summary(state: AppState = Depends(require_succession)) -> List[NineBoxSummary]:
    frame = state.succession_engine.get_9box_summary()
    return [
        NineBoxSummary(category=row['Category'], count=int(row['Count']), percentage=float(row['Percentage']))
        for _, row in frame.iterrows()
    ]


@router.get("/summary")
async def get_succession_summary(state: AppState = Depends(require_succession)) -> Dict[str, Any]:
    analysis = state.succession_engine.analyze_all()

    def safe_serialize(obj):
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        if isinstance(obj, (list, dict, int, float, str, bool, type(None))):
            return obj
        return str(obj)

    readiness = analysis.get('readiness_scores')
    ready_now_count = 0
    if isinstance(readiness, pd.DataFrame):
        ready_now_count = int((readiness['ReadinessLevel'] == 'Ready Now').sum()) if 'ReadinessLevel' in readiness.columns else 0

    return {
        'total_employees': int(len(state.succession_engine.df)),
        'aggregate_ready_now_count': ready_now_count,
        'critical_gap_count': len(analysis.get('critical_gaps', [])),
        'nine_box_summary': safe_serialize(analysis.get('nine_box_summary')),
        'bench_strength': safe_serialize(analysis.get('bench_strength')),
        'governance': 'Aggregate heuristic summaries only; no individual ranking or automated succession decisions.',
    }
