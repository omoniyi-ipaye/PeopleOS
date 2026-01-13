"""
API routes for Employee Experience endpoints.
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Optional, List

from api.dependencies import get_app_state, AppState
from api.schemas.experience import (
    ExperienceIndexResponse,
    EmployeeExperienceResponse,
    SegmentsResponse,
    DriversResponse,
    AtRiskResponse,
    LifecycleResponse,
    ManagerImpactResponse,
    SignalsResponse,
    ExperienceAnalysisResponse,
    GroupExperience,
    EngagementSegment,
    ExperienceDriver,
    AtRiskExperience,
    AtRiskByDepartment,
    LifecycleStage,
    ManagerStats,
    ExperienceSummary,
)

router = APIRouter(prefix="/api/experience", tags=["experience"])


def require_experience(state: AppState = Depends(get_app_state)) -> AppState:
    """Dependency that requires experience engine to be available."""
    if not state.has_data():
        if not state.load_from_database():
            raise HTTPException(
                status_code=400,
                detail="No data loaded. Please upload a file first."
            )

    if state.experience_engine is None:
        raise HTTPException(
            status_code=400,
            detail="Experience analysis not available. Engine initialization may have failed."
        )

    return state


@router.get("/analysis", response_model=ExperienceAnalysisResponse)
async def get_experience_analysis(
    state: AppState = Depends(require_experience)
) -> ExperienceAnalysisResponse:
    """
    Get complete experience analysis.

    Includes:
    - Experience Index (EXI) calculation
    - Engagement segmentation
    - Experience drivers
    - At-risk employees
    - Lifecycle analysis
    - Manager impact
    - Available signals
    - Recommendations
    """
    results = state.experience_engine.analyze_all()

    return ExperienceAnalysisResponse(
        experience_index=ExperienceIndexResponse(**results.get('experience_index', {'available': False})),
        segments=SegmentsResponse(**results.get('segments', {'available': False})),
        drivers=DriversResponse(**results.get('drivers', {'available': False})),
        at_risk=AtRiskResponse(**results.get('at_risk', {'available': False})),
        lifecycle=LifecycleResponse(**results.get('lifecycle', {'available': False})),
        manager_impact=ManagerImpactResponse(**results.get('manager_impact', {'available': False})),
        signals=SignalsResponse(**results.get('signals', {})),
        summary=ExperienceSummary(**results.get('summary', {})),
        recommendations=results.get('recommendations', []),
        warnings=results.get('warnings', [])
    )


@router.get("/index", response_model=ExperienceIndexResponse)
async def get_experience_index(
    group_by: Optional[str] = Query(
        default=None,
        description="Column to segment by (e.g., 'Dept', 'Location')"
    ),
    state: AppState = Depends(require_experience)
) -> ExperienceIndexResponse:
    """
    Get Employee Experience Index (EXI).

    Returns overall EXI score (0-100) with optional segmentation.
    EXI is calculated from available experience signals:
    - eNPS scores
    - Onboarding survey scores
    - Pulse survey scores
    - Manager satisfaction
    - Engagement scores
    - Work-life balance
    - Career growth satisfaction
    """
    results = state.experience_engine.calculate_experience_index(group_by=group_by)
    return ExperienceIndexResponse(**results)


@router.get("/index/employee/{employee_id}", response_model=EmployeeExperienceResponse)
async def get_employee_experience(
    employee_id: str,
    state: AppState = Depends(require_experience)
) -> EmployeeExperienceResponse:
    """
    Get experience details for a specific employee.

    Returns:
    - EXI score
    - Engagement segment
    - Component breakdown
    """
    results = state.experience_engine.get_employee_exi(employee_id)

    if not results.get('available', False):
        raise HTTPException(
            status_code=404,
            detail=results.get('reason', f"Employee {employee_id} not found")
        )

    return EmployeeExperienceResponse(**results)


@router.get("/segments", response_model=SegmentsResponse)
async def get_engagement_segments(
    state: AppState = Depends(require_experience)
) -> SegmentsResponse:
    """
    Get workforce engagement segmentation.

    Segments employees into:
    - Thriving (EXI 80-100): Highly engaged advocates
    - Content (EXI 60-79): Satisfied employees
    - Neutral (EXI 40-59): Neither engaged nor disengaged
    - Disengaged (EXI 20-39): At-risk, showing warning signs
    - Critical (EXI 0-19): Immediate intervention needed
    """
    results = state.experience_engine.get_engagement_segments()
    return SegmentsResponse(**results)


@router.get("/drivers", response_model=DriversResponse)
async def get_experience_drivers(
    state: AppState = Depends(require_experience)
) -> DriversResponse:
    """
    Identify what factors drive experience scores.

    Returns correlation analysis showing which factors
    have the most impact on EXI scores.
    """
    results = state.experience_engine.identify_experience_drivers()
    return DriversResponse(**results)


@router.get("/at-risk", response_model=AtRiskResponse)
async def get_at_risk_employees(
    threshold: Optional[float] = Query(
        default=None,
        description="EXI threshold for at-risk (default: 40)"
    ),
    limit: int = Query(
        default=20,
        ge=1,
        le=100,
        description="Maximum employees to return"
    ),
    state: AppState = Depends(require_experience)
) -> AtRiskResponse:
    """
    Get employees with low experience scores.

    Returns employees below the EXI threshold with:
    - Risk factors
    - Recommended actions
    - Department breakdown
    """
    results = state.experience_engine.get_at_risk_employees(
        threshold=threshold,
        limit=limit
    )
    return AtRiskResponse(**results)


@router.get("/lifecycle", response_model=LifecycleResponse)
async def get_lifecycle_experience(
    state: AppState = Depends(require_experience)
) -> LifecycleResponse:
    """
    Analyze experience by employee lifecycle stage.

    Stages:
    - New Hire (0-6 months)
    - Ramping (6-12 months)
    - Established (1-3 years)
    - Veteran (3+ years)

    Identifies patterns like experience drops at specific tenure points.
    """
    results = state.experience_engine.get_lifecycle_experience()
    return LifecycleResponse(**results)


@router.get("/manager-impact", response_model=ManagerImpactResponse)
async def get_manager_impact(
    state: AppState = Depends(require_experience)
) -> ManagerImpactResponse:
    """
    Analyze how managers affect team experience.

    Requires ManagerID column in data.
    Returns managers with highest and lowest team EXI scores.
    """
    results = state.experience_engine.analyze_manager_impact()
    return ManagerImpactResponse(**results)


@router.get("/signals", response_model=SignalsResponse)
async def get_available_signals(
    state: AppState = Depends(require_experience)
) -> SignalsResponse:
    """
    Get available experience signals in the data.

    Reports which experience columns are present:
    - eNPS_Score
    - Onboarding_30d/60d/90d
    - Pulse_Score
    - ManagerSatisfaction
    - EngagementScore
    - WorkLifeBalance
    - CareerGrowthSatisfaction

    Also provides recommendations for improving data coverage.
    """
    results = state.experience_engine.get_available_signals()
    return SignalsResponse(**results)


@router.get("/trends")
async def get_experience_trends(
    period: str = Query(
        default="month",
        description="Trend period: 'week', 'month', or 'quarter'"
    ),
    state: AppState = Depends(require_experience)
):
    """
    Get EXI trends over time.

    Note: Requires historical data with date columns.
    Currently returns snapshot analysis if no time series available.
    """
    # For now, return current snapshot
    # Future: implement time series tracking
    results = state.experience_engine.calculate_experience_index()

    return {
        'available': results.get('available', False),
        'current_exi': results.get('overall_exi'),
        'period': period,
        'message': 'Time series trends require historical data tracking. '
                   'Current snapshot provided.',
        'trends': []
    }
