"""
API routes for Structural Analysis endpoints.
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Optional, List

from api.dependencies import get_app_state, AppState
from api.schemas.structural import (
    StagnationEmployee,
    StagnationHotspotsResponse,
    ManagerSpan,
    SpanOfControlResponse,
    PromotionEquityResponse,
    PromotionBottlenecksResponse,
    StructuralAnalysisResponse,
    StructuralAnalysisSummary,
)

router = APIRouter(prefix="/api/structural", tags=["structural"])


def require_structural(state: AppState = Depends(get_app_state)) -> AppState:
    """Dependency that requires structural engine to be available."""
    if not state.has_data():
        if not state.load_from_database():
            raise HTTPException(
                status_code=400,
                detail="No data loaded. Please upload a file first."
            )

    if state.structural_engine is None:
        raise HTTPException(
            status_code=400,
            detail="Structural analysis not available. Ensure data has required columns (Tenure, YearsInCurrentRole, ManagerID)."
        )

    return state


@router.get("/analysis", response_model=StructuralAnalysisResponse)
async def get_structural_analysis(
    state: AppState = Depends(require_structural)
) -> StructuralAnalysisResponse:
    """
    Get complete structural analysis.

    Includes:
    - Role stagnation metrics
    - Span of control analysis
    - Promotion equity audit
    - Promotion bottlenecks
    """
    results = state.structural_engine.analyze_all()

    return StructuralAnalysisResponse(
        stagnation=results.get('stagnation', {}),
        span_of_control=results.get('span_of_control', {}),
        promotion_equity=results.get('promotion_equity', {}),
        promotion_bottlenecks=results.get('promotion_bottlenecks', {}),
        summary=StructuralAnalysisSummary(**results.get('summary', {})),
        recommendations=results.get('recommendations', []),
        warnings=results.get('warnings', [])
    )


@router.get("/stagnation", response_model=List[StagnationEmployee])
async def get_stagnation_index(
    dept: Optional[str] = Query(default=None, description="Filter by department"),
    category: Optional[str] = Query(
        default=None,
        description="Filter by category (Critical, Warning, Monitor, Healthy)"
    ),
    min_tenure: Optional[float] = Query(
        default=None,
        ge=0,
        description="Minimum tenure in years"
    ),
    limit: int = Query(default=50, ge=1, le=500),
    state: AppState = Depends(require_structural)
) -> List[StagnationEmployee]:
    """
    Get stagnation index for employees.

    Stagnation Index = YearsInCurrentRole / Tenure
    Higher values indicate longer time in same role relative to tenure.
    """
    stagnation_df = state.structural_engine.calculate_stagnation_index()

    if stagnation_df.empty:
        return []

    # Apply filters
    if dept:
        stagnation_df = stagnation_df[stagnation_df['Dept'] == dept]

    if category:
        stagnation_df = stagnation_df[stagnation_df['StagnationCategory'] == category]

    if min_tenure is not None:
        stagnation_df = stagnation_df[stagnation_df['Tenure'] >= min_tenure]

    # Sort by stagnation index descending
    stagnation_df = stagnation_df.sort_values('StagnationIndex', ascending=False).head(limit)

    return [StagnationEmployee(**row.to_dict()) for _, row in stagnation_df.iterrows()]


@router.get("/stagnation/hotspots", response_model=StagnationHotspotsResponse)
async def get_stagnation_hotspots(
    state: AppState = Depends(require_structural)
) -> StagnationHotspotsResponse:
    """
    Identify departments and job levels with high stagnation.

    Returns areas where multiple employees show signs of role stagnation.
    """
    results = state.structural_engine.identify_stagnation_hotspots()

    if not results.get('available', False):
        return StagnationHotspotsResponse(
            available=False,
            reason=results.get('reason', 'Analysis not available')
        )

    from api.schemas.structural import StagnationHotspot, StagnationSummary

    return StagnationHotspotsResponse(
        available=True,
        hotspots=[StagnationHotspot(**h) for h in results.get('hotspots', [])],
        critical_employees=results.get('critical_employees', []),
        summary=StagnationSummary(**results.get('summary', {})) if results.get('summary') else None
    )


@router.get("/span-of-control", response_model=List[ManagerSpan])
async def get_span_of_control(
    category: Optional[str] = Query(
        default=None,
        description="Filter by category (Under-Leveraged, Optimal, Stretched, Overloaded, Critical)"
    ),
    dept: Optional[str] = Query(default=None, description="Filter by department"),
    min_reports: Optional[int] = Query(default=None, ge=0),
    limit: int = Query(default=50, ge=1, le=500),
    state: AppState = Depends(require_structural)
) -> List[ManagerSpan]:
    """
    Get span of control (direct reports) for managers.

    Categories:
    - Under-Leveraged: <4 direct reports
    - Optimal: 4-8 direct reports
    - Stretched: 9-12 direct reports
    - Overloaded: 13-15 direct reports
    - Critical: >15 direct reports
    """
    span_df = state.structural_engine.calculate_span_of_control()

    if span_df.empty:
        return []

    # Apply filters
    if category:
        span_df = span_df[span_df['SpanCategory'] == category]

    if dept and 'Dept' in span_df.columns:
        span_df = span_df[span_df['Dept'] == dept]

    if min_reports is not None:
        span_df = span_df[span_df['DirectReports'] >= min_reports]

    span_df = span_df.head(limit)

    return [ManagerSpan(**row.to_dict()) for _, row in span_df.iterrows()]


@router.get("/span-of-control/analysis", response_model=SpanOfControlResponse)
async def get_span_analysis(
    state: AppState = Depends(require_structural)
) -> SpanOfControlResponse:
    """
    Get comprehensive span of control analysis with burnout risk.

    Identifies managers at risk of burnout due to excessive direct reports.
    """
    results = state.structural_engine.analyze_manager_burnout_risk()

    if not results.get('available', False):
        return SpanOfControlResponse(
            available=False,
            reason=results.get('reason', 'Analysis not available')
        )

    from api.schemas.structural import DepartmentSpanSummary, SpanOfControlSummary

    return SpanOfControlResponse(
        available=True,
        at_risk_managers=results.get('at_risk_managers', []),
        department_summary=[
            DepartmentSpanSummary(**d) for d in results.get('department_summary', [])
        ],
        summary=SpanOfControlSummary(**results.get('summary', {})) if results.get('summary') else None,
        recommendations=results.get('recommendations', [])
    )


@router.get("/promotion-equity", response_model=PromotionEquityResponse)
async def get_promotion_equity_audit(
    state: AppState = Depends(require_structural)
) -> PromotionEquityResponse:
    """
    Audit promotion velocity for equity across protected groups.

    Identifies if certain demographic groups wait longer for promotion
    after controlling for tenure, rating, and job level.
    """
    results = state.structural_engine.audit_promotion_velocity()

    if not results.get('available', False):
        return PromotionEquityResponse(
            available=False,
            reason=results.get('reason', 'Analysis not available')
        )

    from api.schemas.structural import EquityAuditResult, EquityAuditSummary, PromotionGap

    audit_results = []
    for r in results.get('audit_results', []):
        audit_results.append(EquityAuditResult(
            attribute=r['attribute'],
            reference_group=r['reference_group'],
            group_statistics=r['group_statistics'],
            gaps=[PromotionGap(**g) for g in r.get('gaps', [])],
            p_value=r['p_value'],
            significant_gap=r['significant_gap'],
            finding=r['finding']
        ))

    significant_gaps = [r for r in audit_results if r.significant_gap]

    return PromotionEquityResponse(
        available=True,
        audit_results=audit_results,
        significant_gaps=significant_gaps,
        summary=EquityAuditSummary(**results.get('summary', {})) if results.get('summary') else None,
        recommendations=results.get('recommendations', []),
        methodology=results.get('methodology')
    )


@router.get("/promotion-bottlenecks", response_model=PromotionBottlenecksResponse)
async def get_promotion_bottlenecks(
    state: AppState = Depends(require_structural)
) -> PromotionBottlenecksResponse:
    """
    Identify promotion bottlenecks by department and job level.

    Finds areas where employees wait significantly longer than average for promotion.
    """
    results = state.structural_engine.get_promotion_bottlenecks()

    if not results.get('available', False):
        return PromotionBottlenecksResponse(
            available=False,
            reason=results.get('reason', 'Analysis not available')
        )

    from api.schemas.structural import PromotionBottleneck, BottleneckSummary

    return PromotionBottlenecksResponse(
        available=True,
        bottlenecks=[PromotionBottleneck(**b) for b in results.get('bottlenecks', [])],
        employees_waiting_longest=results.get('employees_waiting_longest', []),
        summary=BottleneckSummary(**results.get('summary', {})) if results.get('summary') else None
    )


@router.get("/employee/{employee_id}/stagnation")
async def get_employee_stagnation(
    employee_id: str,
    state: AppState = Depends(require_structural)
):
    """
    Get stagnation metrics for a specific employee.
    """
    stagnation_df = state.structural_engine.calculate_stagnation_index()

    if stagnation_df.empty:
        raise HTTPException(
            status_code=400,
            detail="Stagnation analysis not available"
        )

    employee_data = stagnation_df[stagnation_df['EmployeeID'] == employee_id]

    if employee_data.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Employee {employee_id} not found"
        )

    row = employee_data.iloc[0]
    return StagnationEmployee(**row.to_dict())
