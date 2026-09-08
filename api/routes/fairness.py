"""Fairness analysis route handlers."""

from typing import List, Dict, Any

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from api.dependencies import get_app_state, AppState

from src.serialization import json_safe

router = APIRouter(prefix="/api/fairness", tags=["fairness"])


class FourFifthsResult(BaseModel):
    """Four-fifths rule analysis result."""
    attribute: str
    group: str
    selection_rate: float
    reference_rate: float
    ratio: float | None
    passes_rule: bool | None
    status: str
    metric_semantics: str = "favorable_retained_share_screening_not_compliance_determination"


class FairnessAnalysisResponse(BaseModel):
    """Full fairness analysis response."""
    four_fifths: List[FourFifthsResult]
    overall_status: str
    recommendations: List[str]
    warnings: List[str]


def require_fairness(state: AppState = Depends(get_app_state)) -> AppState:
    """Dependency that requires fairness engine."""
    if not state.has_data():
        if not state.load_from_database():
            raise HTTPException(
                status_code=400,
                detail="No data loaded. Please upload a file first."
            )

    if state.fairness_engine is None:
        raise HTTPException(
            status_code=400,
            detail="Fairness analysis requires predictions. Upload data with Attrition column."
        )

    return state


@router.get("/four-fifths", response_model=List[FourFifthsResult])
async def get_four_fifths_analysis(
    state: AppState = Depends(require_fairness)
) -> List[FourFifthsResult]:
    """
    Get descriptive favorable-outcome ratio screening; no compliance determination.
    """
    analysis_df = state.fairness_engine.calculate_four_fifths_rule('Attrition', favorable=False)

    if analysis_df.empty:
        return []

    results = []
    for _, row in analysis_df.iterrows():
        results.append(FourFifthsResult(
            attribute=row['attribute'],
            group=str(row['group']),
            selection_rate=float(row['favorable_rate']),
            reference_rate=float(row['reference_favorable_rate']),
            ratio=json_safe(row['adverse_impact_ratio']),
            passes_rule=json_safe(row['passes_4_5_rule']),
            status='Unavailable' if json_safe(row['passes_4_5_rule']) is None else ('No screening signal' if row['passes_4_5_rule'] else 'Screening signal')
        ))

    return results


@router.get("/analysis", response_model=FairnessAnalysisResponse)
async def get_fairness_analysis(
    state: AppState = Depends(require_fairness)
) -> FairnessAnalysisResponse:
    """
    Get full fairness analysis including four-fifths rule and recommendations.
    """
    summary = state.fairness_engine.get_fairness_summary('Attrition')

    # Get four-fifths results
    four_fifths_df = state.fairness_engine.calculate_four_fifths_rule('Attrition', favorable=False)
    four_fifths = []

    if not four_fifths_df.empty:
        for _, row in four_fifths_df.iterrows():
            four_fifths.append(FourFifthsResult(
                attribute=row['attribute'],
                group=str(row['group']),
                selection_rate=float(row['favorable_rate']),
                reference_rate=float(row['reference_favorable_rate']),
                ratio=json_safe(row['adverse_impact_ratio']),
                passes_rule=json_safe(row['passes_4_5_rule']),
                status='Unavailable' if json_safe(row['passes_4_5_rule']) is None else ('No screening signal' if row['passes_4_5_rule'] else 'Screening signal')
            ))

    return FairnessAnalysisResponse(
        four_fifths=four_fifths,
        overall_status=summary.get('overall_status', 'Unknown'),
        recommendations=summary.get('recommendations', []),
        warnings=summary.get('issues_found', [])
    )


@router.get("/demographic-parity")
async def get_demographic_parity(
    state: AppState = Depends(require_fairness)
) -> Dict[str, Any]:
    """
    Get demographic parity analysis across protected attributes.
    """
    parity_df = state.fairness_engine.calculate_demographic_parity('Attrition')

    if parity_df.empty:
        return {'results': [], 'message': 'No demographic parity data available'}

    results = []
    for _, row in parity_df.iterrows():
        results.append({
            'attribute': row['attribute'],
            'group': str(row['group']),
            'rate': float(row['rate']),
            'count': int(row['count']),
            'disparity': float(row['disparity']),
            'parity_ratio': json_safe(row['parity_ratio'])
        })

    return {'results': results}
