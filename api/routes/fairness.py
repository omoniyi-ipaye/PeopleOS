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
    group_size: int | None = None
    suppressed_group_count: int = 0
    metric_semantics: str = "favorable_retained_share_screening_not_compliance_determination"


class FairnessAnalysisResponse(BaseModel):
    """Full fairness analysis response."""
    four_fifths: List[FourFifthsResult]
    overall_status: str
    recommendations: List[str]
    warnings: List[str]
    interpretation_boundary: str = "Fairness metrics are descriptive screening signals, not legal, causal, or bias determinations."


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
            detail="Recorded outcome disparity analysis requires Attrition and protected-group data."
        )

    return state


def _four_fifths_row(row) -> FourFifthsResult:
    passes = json_safe(row['passes_4_5_rule'])
    return FourFifthsResult(
        attribute=row['attribute'],
        group=str(row['group']),
        selection_rate=float(row['favorable_rate']),
        reference_rate=float(row['reference_favorable_rate']),
        ratio=json_safe(row['adverse_impact_ratio']),
        passes_rule=passes,
        status='Unavailable' if passes is None else ('No screening signal' if passes else 'Screening signal'),
        group_size=int(row['count']) if row.get('count') is not None else None,
        suppressed_group_count=int(row.get('suppressed_group_count', 0) or 0),
    )


@router.get("/four-fifths", response_model=List[FourFifthsResult])
async def get_four_fifths_analysis(
    state: AppState = Depends(require_fairness)
) -> List[FourFifthsResult]:
    """Get descriptive favorable-outcome ratio screening; no compliance determination."""
    analysis_df = state.fairness_engine.calculate_four_fifths_rule('Attrition', favorable=False)
    if analysis_df.empty:
        return []
    return [_four_fifths_row(row) for _, row in analysis_df.iterrows()]


@router.get("/analysis", response_model=FairnessAnalysisResponse)
async def get_fairness_analysis(
    state: AppState = Depends(require_fairness)
) -> FairnessAnalysisResponse:
    """Get full fairness screening with explicit interpretation limits."""
    summary = state.fairness_engine.get_fairness_summary('Attrition')
    four_fifths_df = state.fairness_engine.calculate_four_fifths_rule('Attrition', favorable=False)
    four_fifths = [_four_fifths_row(row) for _, row in four_fifths_df.iterrows()] if not four_fifths_df.empty else []

    return FairnessAnalysisResponse(
        four_fifths=four_fifths,
        overall_status=summary.get('overall_status', 'Unknown'),
        recommendations=summary.get('recommendations', []),
        warnings=summary.get('issues_found', []),
        interpretation_boundary=summary.get(
            'interpretation_boundary',
            'Fairness metrics are descriptive screening signals, not legal, causal, or bias determinations.',
        ),
    )


@router.get("/demographic-parity")
async def get_demographic_parity(
    state: AppState = Depends(require_fairness)
) -> Dict[str, Any]:
    """Get observed outcome-rate disparity screening across eligible groups.

    The historical `parity_ratio` field is retained for client compatibility. Its
    authoritative meaning is `outcome_rate_ratio_to_overall`: group observed
    attrition rate divided by the overall known-outcome attrition rate. It is not
    the four-fifths favorable-outcome ratio and is not a fairness determination.
    """
    parity_df = state.fairness_engine.calculate_demographic_parity('Attrition')

    if parity_df.empty:
        return {
            'results': [],
            'message': 'No eligible outcome-disparity data available; absence of results is not evidence of parity.',
            'metric_semantics': 'observed_attrition_rate_disparity_not_fairness_determination',
        }

    results = []
    for _, row in parity_df.iterrows():
        ratio = json_safe(row.get('outcome_rate_ratio_to_overall', row.get('parity_ratio')))
        results.append({
            'attribute': row['attribute'],
            'dimension_type': row.get('dimension_type'),
            'group': str(row['group']),
            'rate': float(row['rate']),
            'count': int(row['count']),
            'disparity': json_safe(row.get('disparity')),
            'outcome_rate_ratio_to_overall': ratio,
            'parity_ratio': ratio,
            'overall_known_outcome_count': int(row.get('overall_known_outcome_count', 0) or 0),
            'attribute_observed_count': int(row.get('attribute_observed_count', 0) or 0),
            'attribute_coverage': json_safe(row.get('attribute_coverage')),
            'suppressed_group_count': int(row.get('suppressed_group_count', 0) or 0),
            'metric_semantics': 'observed_attrition_rate_disparity_not_fairness_determination',
        })

    return {
        'results': results,
        'metric_semantics': 'observed_attrition_rate_disparity_not_fairness_determination',
        'interpretation_boundary': 'Use favorable-outcome four-fifths results for that specific screening ratio; neither endpoint establishes discrimination, fairness, or causation.',
    }
