"""Governed Employee Experience API routes.

Experience is measured from explicit survey/experience signals. PeopleOS does not
infer employee sentiment from tenure, performance, salary or promotion proxies,
and does not expose individual experience rankings through the enterprise API.
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from api.dependencies import AppState, get_app_state
from api.schemas.experience import (
    AtRiskResponse,
    DriversResponse,
    ExperienceAnalysisResponse,
    ExperienceIndexResponse,
    ExperienceSummary,
    LifecycleResponse,
    ManagerImpactResponse,
    SegmentsResponse,
    SignalsResponse,
)

router = APIRouter(prefix="/api/experience", tags=["experience"])


def require_experience(state: AppState = Depends(get_app_state)) -> AppState:
    if not state.has_data():
        if not state.load_from_database():
            raise HTTPException(status_code=400, detail="No data loaded. Please upload a file first.")
    if state.experience_engine is None:
        raise HTTPException(status_code=400, detail="Experience analysis is unavailable for the current dataset.")
    return state


def _has_measured_signals(state: AppState) -> bool:
    engine = state.experience_engine
    return bool(engine is not None and int(getattr(engine, "available_survey_signals", 0)) > 0)


def _signals(state: AppState) -> SignalsResponse:
    engine = state.experience_engine
    raw = engine.get_available_signals() if engine is not None else {}
    return SignalsResponse(**raw)


def _unavailable_index(state: AppState) -> ExperienceIndexResponse:
    return ExperienceIndexResponse(
        available=False,
        reason="No measured experience survey signals are available. PeopleOS will not infer experience from HRIS proxy fields.",
        total_employees=len(state.raw_df) if state.raw_df is not None else 0,
        signals_available=int(getattr(state.experience_engine, "available_survey_signals", 0) or 0),
    )


def _safe_index(state: AppState, group_by: Optional[str] = None) -> ExperienceIndexResponse:
    if not _has_measured_signals(state):
        return _unavailable_index(state)
    raw = state.experience_engine.calculate_experience_index(group_by=group_by)
    raw["benchmark"] = None
    if raw.get("available"):
        raw["interpretation"] = "Configured weighted composite of available measured experience signals. Interpret together with signal coverage and component definitions."
        for group in raw.get("by_group", []) or []:
            group["interpretation"] = "Group mean of the configured measured-signal composite."
    return ExperienceIndexResponse(**raw)


def _safe_drivers(state: AppState) -> DriversResponse:
    if not _has_measured_signals(state):
        return DriversResponse(available=False, reason="Measured experience signals are required for association analysis.")
    raw = state.experience_engine.identify_experience_drivers()
    if not raw.get("available", False):
        return DriversResponse(**raw)
    raw["recommendations"] = [
        "Treat these as observed associations with the configured experience composite. Validate direction, confounding and stability before changing policy or manager practice."
    ]
    return DriversResponse(**raw)


def _safe_segments(state: AppState) -> SegmentsResponse:
    if not _has_measured_signals(state):
        return SegmentsResponse(available=False, reason="Measured experience signals are required before segmenting the experience index.")
    raw = state.experience_engine.get_engagement_segments()
    if raw.get("available", False):
        raw["recommendations"] = [
            "Segments are configured score bands for aggregate monitoring; they are not diagnoses of individual engagement. Suppressed cells must not be reconstructed from totals."
        ]
    return SegmentsResponse(**raw)


def _safe_lifecycle(state: AppState) -> LifecycleResponse:
    if not _has_measured_signals(state):
        return LifecycleResponse(available=False, reason="Measured experience signals are required for lifecycle experience comparison.")
    raw = state.experience_engine.get_lifecycle_experience()
    if raw.get("available", False):
        raw["recommendations"] = [
            "Lifecycle differences are descriptive associations. Compare sample sizes and survey coverage before interpreting them as stage effects."
        ]
    return LifecycleResponse(**raw)


def _safe_at_risk(state: AppState, threshold: Optional[float] = None) -> AtRiskResponse:
    if not _has_measured_signals(state):
        return AtRiskResponse(available=False, reason="Measured experience signals are required for low-score aggregate monitoring.")
    raw = state.experience_engine.get_at_risk_employees(threshold=threshold, limit=1000)
    if not raw.get("available", False):
        return AtRiskResponse(**raw)
    return AtRiskResponse(
        available=True,
        total_at_risk=raw.get("total_at_risk"),
        threshold_used=raw.get("threshold_used"),
        employees=None,
        by_department=raw.get("by_department"),
        suppressed=bool(raw.get("suppressed", False)),
        metric_semantics=raw.get("metric_semantics"),
    )


@router.get("/analysis", response_model=ExperienceAnalysisResponse)
async def get_experience_analysis(state: AppState = Depends(require_experience)) -> ExperienceAnalysisResponse:
    signals = _signals(state)
    if not _has_measured_signals(state):
        return ExperienceAnalysisResponse(
            experience_index=_unavailable_index(state),
            segments=SegmentsResponse(available=False, reason="Measured experience signals are unavailable."),
            drivers=DriversResponse(available=False, reason="Measured experience signals are unavailable."),
            at_risk=AtRiskResponse(available=False, reason="Measured experience signals are unavailable."),
            lifecycle=LifecycleResponse(available=False, reason="Measured experience signals are unavailable."),
            manager_impact=ManagerImpactResponse(available=False, reason="Manager ranking is outside the governed aggregate experience boundary."),
            signals=signals,
            summary=ExperienceSummary(
                overall_exi=None,
                health_indicator="Unavailable",
                total_employees=len(state.raw_df) if state.raw_df is not None else 0,
                at_risk_count=None,
                signals_available=signals.total_signals,
                total_warnings=1,
                total_recommendations=1,
            ),
            recommendations=["Add explicit experience survey signals before interpreting workforce experience."],
            warnings=["PeopleOS did not derive an experience score from tenure, performance, salary or promotion proxies."],
        )

    index = _safe_index(state)
    segments = _safe_segments(state)
    drivers = _safe_drivers(state)
    at_risk = _safe_at_risk(state)
    lifecycle = _safe_lifecycle(state)
    return ExperienceAnalysisResponse(
        experience_index=index,
        segments=segments,
        drivers=drivers,
        at_risk=at_risk,
        lifecycle=lifecycle,
        manager_impact=ManagerImpactResponse(available=False, reason="Manager-level experience ranking is disabled in the governed product boundary."),
        signals=signals,
        summary=ExperienceSummary(
            overall_exi=index.overall_exi,
            health_indicator=segments.health_indicator or "Measured",
            total_employees=index.total_employees or 0,
            at_risk_count=None if at_risk.suppressed else at_risk.total_at_risk,
            signals_available=signals.total_signals,
            total_warnings=0,
            total_recommendations=1,
        ),
        recommendations=["Use measured experience signals for aggregate investigation; validate associations before intervention."],
        warnings=["Experience Index is a configurable weighted composite, not an externally validated benchmark."],
    )


@router.get("/index", response_model=ExperienceIndexResponse)
async def get_experience_index(
    group_by: Optional[str] = Query(default=None, description="Optional aggregate grouping such as Dept or Location"),
    state: AppState = Depends(require_experience),
) -> ExperienceIndexResponse:
    return _safe_index(state, group_by=group_by)


@router.get("/index/employee/{employee_id}", deprecated=True)
async def get_employee_experience(employee_id: str, state: AppState = Depends(require_experience)):
    raise HTTPException(status_code=403, detail="Individual experience scoring is disabled. Use aggregate experience analysis.")


@router.get("/segments", response_model=SegmentsResponse)
async def get_engagement_segments(state: AppState = Depends(require_experience)) -> SegmentsResponse:
    return _safe_segments(state)


@router.get("/drivers", response_model=DriversResponse)
async def get_experience_drivers(state: AppState = Depends(require_experience)) -> DriversResponse:
    return _safe_drivers(state)


@router.get("/at-risk", response_model=AtRiskResponse)
async def get_at_risk_employees(
    threshold: Optional[float] = Query(default=None, description="Configured composite threshold for aggregate monitoring"),
    state: AppState = Depends(require_experience),
) -> AtRiskResponse:
    return _safe_at_risk(state, threshold=threshold)


@router.get("/lifecycle", response_model=LifecycleResponse)
async def get_lifecycle_experience(state: AppState = Depends(require_experience)) -> LifecycleResponse:
    return _safe_lifecycle(state)


@router.get("/manager-impact", response_model=ManagerImpactResponse, deprecated=True)
async def get_manager_impact(state: AppState = Depends(require_experience)) -> ManagerImpactResponse:
    return ManagerImpactResponse(available=False, reason="Manager-level experience ranking is disabled in the governed product boundary.")


@router.get("/signals", response_model=SignalsResponse)
async def get_available_signals(state: AppState = Depends(require_experience)) -> SignalsResponse:
    return _signals(state)


@router.get("/trends")
async def get_experience_trends(
    period: str = Query(default="month", description="Requested trend period"),
    state: AppState = Depends(require_experience),
):
    return {
        "available": False,
        "current_exi": _safe_index(state).overall_exi if _has_measured_signals(state) else None,
        "period": period,
        "message": "Time-series experience trends require repeated dated experience measurements. A current snapshot is not a trend.",
        "trends": [],
    }
