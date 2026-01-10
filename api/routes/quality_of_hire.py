"""
API routes for Quality of Hire endpoints.
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Optional, List

from api.dependencies import get_app_state, AppState
from api.schemas.quality_of_hire import (
    QualityOfHireAnalysisResponse,
    SourceEffectiveness,
    CorrelationAnalysisResponse,
    PrehireCorrelation,
    HiringInsights,
    CohortPerformance,
    NewHireRisk,
    QualityOfHireSummary,
)

router = APIRouter(prefix="/api/quality-of-hire", tags=["quality_of_hire"])


def require_quality_of_hire(state: AppState = Depends(get_app_state)) -> AppState:
    """Dependency that requires quality of hire engine to be available."""
    if not state.has_data():
        state.load_from_database()
            
    return state


@router.get("/analysis", response_model=QualityOfHireAnalysisResponse)
async def get_quality_of_hire_analysis(
    state: AppState = Depends(require_quality_of_hire)
) -> QualityOfHireAnalysisResponse:
    """Get complete quality of hire analysis."""
    if state.quality_of_hire_engine is None:
        return QualityOfHireAnalysisResponse(
            source_effectiveness=[],
            correlations=None,
            retention_correlations=None,
            insights=None,
            cohort_analysis=[],
            new_hire_risks=[],
            summary=QualityOfHireSummary(
                total_employees=len(state.raw_df) if state.raw_df is not None else 0,
                has_hire_source=False,
                has_interview_scores=False,
                has_assessment=False,
                prehire_signals_count=0,
                sources_analyzed=0,
                new_hires_at_risk=0
            ),
            recommendations=[],
            warnings=["Quality of Hire engine not initialized. Check if data contains 'HireSource' or interview columns."]
        )

    results = state.quality_of_hire_engine.analyze_all()
    # Convert correlation results
    correlations = None
    if results.get('correlations', {}).get('available'):
        corr_data = results['correlations']
        correlations = CorrelationAnalysisResponse(
            available=True,
            outcome_column=corr_data.get('outcome_column'),
            correlations=[PrehireCorrelation(**c) for c in corr_data.get('correlations', [])],
            best_predictors=[PrehireCorrelation(**c) for c in corr_data.get('best_predictors', [])],
            non_predictors=[PrehireCorrelation(**c) for c in corr_data.get('non_predictors', [])],
            recommendations=corr_data.get('recommendations', [])
        )

    retention_correlations = None
    if results.get('retention_correlations', {}).get('available'):
        ret_data = results['retention_correlations']
        retention_correlations = CorrelationAnalysisResponse(
            available=True,
            outcome_column=ret_data.get('outcome_column'),
            correlations=[PrehireCorrelation(**c) for c in ret_data.get('correlations', [])],
            best_predictors=[PrehireCorrelation(**c) for c in ret_data.get('best_predictors', [])],
            non_predictors=[PrehireCorrelation(**c) for c in ret_data.get('non_predictors', [])],
            recommendations=ret_data.get('recommendations', [])
        )

    return QualityOfHireAnalysisResponse(
        source_effectiveness=[SourceEffectiveness(**s) for s in results.get('source_effectiveness', [])],
        correlations=correlations,
        retention_correlations=retention_correlations,
        insights=HiringInsights(**results.get('insights', {})) if results.get('insights') else None,
        cohort_analysis=[CohortPerformance(cohort_name=c.get('HireSource', 'Unknown'), **{k: v for k, v in c.items() if k != 'HireSource'}) for c in results.get('cohort_analysis', [])],
        new_hire_risks=[NewHireRisk(**r) for r in results.get('new_hire_risks', [])],
        summary=QualityOfHireSummary(**results.get('summary', {})),
        recommendations=results.get('recommendations', []),
        warnings=results.get('warnings', [])
    )


@router.get("/source-effectiveness", response_model=List[SourceEffectiveness])
async def get_source_effectiveness(
    state: AppState = Depends(require_quality_of_hire)
) -> List[SourceEffectiveness]:
    """
    Get effectiveness metrics for each hiring source.

    Compares sources on:
    - Performance ratings
    - Retention rates
    - Promotion rates
    - Overall quality score
    """
    if state.quality_of_hire_engine is None:
        return []
    source_df = state.quality_of_hire_engine.calculate_source_effectiveness()

    if source_df.empty:
        return []

    return [SourceEffectiveness(**row.to_dict()) for _, row in source_df.iterrows()]


@router.get("/correlations", response_model=CorrelationAnalysisResponse)
async def get_prehire_posthire_correlations(
    outcome: str = Query(
        default="LastRating",
        description="Post-hire outcome to correlate against (e.g., 'LastRating', 'Attrition')"
    ),
    state: AppState = Depends(require_quality_of_hire)
) -> CorrelationAnalysisResponse:
    """
    Get correlations between pre-hire signals and post-hire outcomes.

    Identifies which interview dimensions and assessments predict actual performance.
    """
    if state.quality_of_hire_engine is None:
        return CorrelationAnalysisResponse(
            available=False,
            reason="Quality of Hire engine not initialized"
        )
    results = state.quality_of_hire_engine.correlate_prehire_posthire(outcome_column=outcome)

    if not results.get('available', False):
        return CorrelationAnalysisResponse(
            available=False,
            reason=results.get('reason', 'Analysis not available')
        )

    return CorrelationAnalysisResponse(
        available=True,
        outcome_column=results.get('outcome_column'),
        correlations=[PrehireCorrelation(**c) for c in results.get('correlations', [])],
        best_predictors=[PrehireCorrelation(**c) for c in results.get('best_predictors', [])],
        non_predictors=[PrehireCorrelation(**c) for c in results.get('non_predictors', [])],
        recommendations=results.get('recommendations', [])
    )


@router.get("/insights", response_model=HiringInsights)
async def get_hiring_insights(
    state: AppState = Depends(require_quality_of_hire)
) -> HiringInsights:
    """
    Get strategic hiring insights.

    Combines source effectiveness and correlation analysis into actionable recommendations.
    """
    if state.quality_of_hire_engine is None:
        raise HTTPException(status_code=404, detail="Hiring insights not available")
    results = state.quality_of_hire_engine.get_hiring_insights()
    return HiringInsights(**results)


@router.get("/cohort-analysis", response_model=List[CohortPerformance])
async def get_cohort_analysis(
    cohort_by: str = Query(
        default="HireSource",
        description="Column to group by (e.g., 'HireSource', 'Dept')"
    ),
    min_tenure_months: int = Query(
        default=6,
        ge=0,
        description="Minimum tenure in months to include"
    ),
    state: AppState = Depends(require_quality_of_hire)
) -> List[CohortPerformance]:
    """
    Analyze performance by cohort.

    Compare how different hiring cohorts perform over time.
    """
    if state.quality_of_hire_engine is None:
        return []
    cohort_df = state.quality_of_hire_engine.analyze_cohort_performance(
        cohort_column=cohort_by,
        min_tenure_months=min_tenure_months
    )

    if cohort_df.empty:
        return []

    results = []
    for _, row in cohort_df.iterrows():
        cohort_name = row.get(cohort_by, 'Unknown')
        row_dict = row.to_dict()
        # Remove the cohort column from dict to avoid duplicate
        if cohort_by in row_dict:
            del row_dict[cohort_by]
        results.append(CohortPerformance(cohort_name=str(cohort_name), **row_dict))

    return results


@router.get("/new-hire-risks", response_model=List[NewHireRisk])
async def get_new_hire_risks(
    months: int = Query(
        default=6,
        ge=1,
        le=24,
        description="Look at hires within this many months"
    ),
    state: AppState = Depends(require_quality_of_hire)
) -> List[NewHireRisk]:
    """
    Get risk assessment for recent hires.

    Identifies new hires who may need additional support based on pre-hire signals.
    """
    if state.quality_of_hire_engine is None:
        return []
    risk_df = state.quality_of_hire_engine.get_new_hire_risk_assessment(
        months_since_hire=months
    )

    if risk_df.empty:
        return []

    return [NewHireRisk(**row.to_dict()) for _, row in risk_df.iterrows()]


@router.get("/best-predictors", response_model=List[PrehireCorrelation])
async def get_best_predictors(
    limit: int = Query(default=5, ge=1, le=20),
    state: AppState = Depends(require_quality_of_hire)
) -> List[PrehireCorrelation]:
    """
    Get the top predictors of post-hire performance.

    Returns the pre-hire signals with strongest correlation to performance.
    """
    if state.quality_of_hire_engine is None:
        return []
    results = state.quality_of_hire_engine.correlate_prehire_posthire()

    if not results.get('available', False):
        return []

    best = results.get('best_predictors', [])[:limit]
    return [PrehireCorrelation(**c) for c in best]
