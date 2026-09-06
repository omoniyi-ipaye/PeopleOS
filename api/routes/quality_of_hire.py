"""Governed Quality of Hire API routes.

The legacy engine contains exploratory heuristics. This route is the product
integrity boundary: outputs are aggregate, observational, and never presented as
causal proof or as a basis for automatic hiring decisions.
"""

from typing import List

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query

from api.dependencies import AppState, get_app_state
from api.schemas.quality_of_hire import (
    CohortPerformance,
    CorrelationAnalysisResponse,
    HiringInsights,
    NewHireRisk,
    PrehireCorrelation,
    QualityOfHireAnalysisResponse,
    QualityOfHireSummary,
    ROIAnalysis,
    RedFlag,
    SourceEffectiveness,
)

router = APIRouter(prefix="/api/quality-of-hire", tags=["quality_of_hire"])


def require_quality_of_hire(state: AppState = Depends(get_app_state)) -> AppState:
    if not state.has_data():
        state.load_from_database()
    return state


def _normalize_correlation_inputs(state: AppState) -> None:
    engine = state.quality_of_hire_engine
    if engine is None:
        return
    columns = set(getattr(engine, "prehire_columns", []))
    columns.update({"LastRating", "Attrition"})
    for column in columns:
        if column in engine.df.columns:
            engine.df[column] = pd.to_numeric(engine.df[column], errors="coerce").astype("float64")


def _safe_correlation(item: dict) -> PrehireCorrelation:
    """Translate an engine correlation into an observational product claim."""
    correlation = float(item.get("correlation", 0.0))
    strength = item.get("strength", "Unknown")
    direction = item.get("direction", "positive" if correlation >= 0 else "negative")
    significant = bool(item.get("is_significant", False))
    display = str(item.get("display_name") or item.get("predictor") or "Signal")
    qualifier = "statistically distinguishable from zero in this sample" if significant else "not statistically distinguishable from zero in this sample"
    interpretation = (
        f"{strength} {direction} association between {display} and "
        f"{item.get('outcome_column', 'the post-hire outcome')}; {qualifier}."
    )
    return PrehireCorrelation(
        predictor=str(item.get("predictor", display)),
        display_name=display,
        correlation=correlation,
        abs_correlation=abs(correlation),
        p_value=float(item.get("p_value", 1.0)),
        is_significant=significant,
        strength=strength,
        direction=direction,
        sample_size=int(item.get("sample_size", 0)),
        interpretation=interpretation,
        insight=None,
    )


def _safe_correlation_response(raw: dict) -> CorrelationAnalysisResponse:
    if not raw.get("available", False):
        return CorrelationAnalysisResponse(available=False, reason=raw.get("reason", "Analysis not available"))
    all_items = []
    for item in raw.get("correlations", []):
        enriched = dict(item)
        enriched["outcome_column"] = raw.get("outcome_column")
        all_items.append(_safe_correlation(enriched))
    strongest = [item for item in all_items if item.is_significant and item.abs_correlation >= 0.2]
    weak = [item for item in all_items if (not item.is_significant) or item.abs_correlation < 0.1]
    return CorrelationAnalysisResponse(
        available=True,
        outcome_column=raw.get("outcome_column"),
        correlations=all_items,
        # Backward-compatible field name; semantics are "strongest observed associations".
        best_predictors=strongest,
        non_predictors=weak,
        recommendations=[
            "Treat these results as observational screening evidence only. Validate promising signals prospectively before changing hiring rubrics or selection weights."
        ],
    )


def _safe_source(row: dict) -> SourceEffectiveness:
    """Remove unsupported automatic investment recommendations from source metrics."""
    payload = dict(row)
    payload["recommendation"] = "Compare with role mix, tenure exposure, cost and future cohorts before changing source allocation."
    return SourceEffectiveness(**payload)


def _safe_insights(raw: dict) -> HiringInsights:
    summary = raw.get("summary", {}) or {}
    sources = [_safe_source(item) for item in raw.get("top_sources", [])]
    top = []
    for item in raw.get("top_predictors", []):
        top.append(_safe_correlation({**item, "outcome_column": "LastRating"}))
    red_flags = [RedFlag(**item) for item in raw.get("red_flags", [])]
    roi = {key: ROIAnalysis(**value) for key, value in (raw.get("roi_analysis", {}) or {}).items()}
    return HiringInsights(
        summary=summary,
        top_sources=sources,
        top_predictors=top,
        red_flags=red_flags,
        recommendations=[
            "Observed source and pre-hire associations are exploratory. Do not infer causation or automatically change hiring criteria without prospective validation."
        ],
        roi_analysis=roi,
    )


@router.get("/analysis", response_model=QualityOfHireAnalysisResponse)
async def get_quality_of_hire_analysis(state: AppState = Depends(require_quality_of_hire)) -> QualityOfHireAnalysisResponse:
    if state.quality_of_hire_engine is None:
        return QualityOfHireAnalysisResponse(
            summary=QualityOfHireSummary(
                total_employees=len(state.raw_df) if state.raw_df is not None else 0,
                has_hire_source=False,
                has_interview_scores=False,
                has_assessment=False,
                prehire_signals_count=0,
                sources_analyzed=0,
                new_hires_at_risk=0,
            ),
            warnings=["Quality of Hire analysis is unavailable because the required hiring-source or pre-hire signal fields are not present."],
        )

    _normalize_correlation_inputs(state)
    results = state.quality_of_hire_engine.analyze_all()
    source_rows = [_safe_source(item) for item in results.get("source_effectiveness", [])]
    correlations = _safe_correlation_response(results.get("correlations", {})) if results.get("correlations") else None
    retention = _safe_correlation_response(results.get("retention_correlations", {})) if results.get("retention_correlations") else None
    cohort = [
        CohortPerformance(cohort_name=item.get("HireSource", "Unknown"), **{k: v for k, v in item.items() if k != "HireSource"})
        for item in results.get("cohort_analysis", [])
    ]
    warnings = list(results.get("warnings", []) or [])
    warnings.append("Quality scores are configurable composite heuristics; they are not validated measures of hire quality or causal source effectiveness.")
    warnings.append("Retention fields represent observed retained share in the available cohort unless a duration-qualified retention window is explicitly available.")
    return QualityOfHireAnalysisResponse(
        source_effectiveness=source_rows,
        correlations=correlations,
        retention_correlations=retention,
        insights=_safe_insights(results.get("insights", {})) if results.get("insights") else None,
        cohort_analysis=cohort,
        # Individual new-hire risk ranking is outside the governed product boundary.
        new_hire_risks=[],
        summary=QualityOfHireSummary(**results.get("summary", {})),
        recommendations=["Use Quality of Hire to generate hypotheses for prospective validation, not automatic selection or sourcing decisions."],
        warnings=warnings,
    )


@router.get("/source-effectiveness", response_model=List[SourceEffectiveness])
async def get_source_effectiveness(state: AppState = Depends(require_quality_of_hire)) -> List[SourceEffectiveness]:
    if state.quality_of_hire_engine is None:
        return []
    frame = state.quality_of_hire_engine.calculate_source_effectiveness()
    return [_safe_source(row.to_dict()) for _, row in frame.iterrows()] if not frame.empty else []


@router.get("/correlations", response_model=CorrelationAnalysisResponse)
async def get_prehire_posthire_correlations(
    outcome: str = Query(default="LastRating", description="Post-hire outcome used for observational association screening"),
    state: AppState = Depends(require_quality_of_hire),
) -> CorrelationAnalysisResponse:
    if state.quality_of_hire_engine is None:
        return CorrelationAnalysisResponse(available=False, reason="Quality of Hire engine not initialized")
    _normalize_correlation_inputs(state)
    return _safe_correlation_response(state.quality_of_hire_engine.correlate_prehire_posthire(outcome_column=outcome))


@router.get("/insights", response_model=HiringInsights)
async def get_hiring_insights(state: AppState = Depends(require_quality_of_hire)) -> HiringInsights:
    if state.quality_of_hire_engine is None:
        raise HTTPException(status_code=404, detail="Hiring insights not available")
    _normalize_correlation_inputs(state)
    return _safe_insights(state.quality_of_hire_engine.get_hiring_insights())


@router.get("/cohort-analysis", response_model=List[CohortPerformance])
async def get_cohort_analysis(
    cohort_by: str = Query(default="HireSource", description="Column to group by"),
    min_tenure_months: int = Query(default=6, ge=0, description="Minimum tenure exposure in months"),
    state: AppState = Depends(require_quality_of_hire),
) -> List[CohortPerformance]:
    if state.quality_of_hire_engine is None:
        return []
    frame = state.quality_of_hire_engine.analyze_cohort_performance(cohort_column=cohort_by, min_tenure_months=min_tenure_months)
    if frame.empty:
        return []
    output = []
    for _, row in frame.iterrows():
        data = row.to_dict()
        cohort_name = str(data.pop(cohort_by, "Unknown"))
        output.append(CohortPerformance(cohort_name=cohort_name, **data))
    return output


@router.get("/new-hire-risks", response_model=List[NewHireRisk], deprecated=True)
async def get_new_hire_risks(state: AppState = Depends(require_quality_of_hire)) -> List[NewHireRisk]:
    """Disabled: individual new-hire ranking is outside PeopleOS aggregate governance."""
    return []


@router.get("/best-predictors", response_model=List[PrehireCorrelation], deprecated=True)
async def get_best_predictors(
    limit: int = Query(default=5, ge=1, le=20),
    state: AppState = Depends(require_quality_of_hire),
) -> List[PrehireCorrelation]:
    """Backward-compatible endpoint returning strongest observed associations, not validated predictors."""
    if state.quality_of_hire_engine is None:
        return []
    _normalize_correlation_inputs(state)
    response = _safe_correlation_response(state.quality_of_hire_engine.correlate_prehire_posthire())
    return response.best_predictors[:limit]
