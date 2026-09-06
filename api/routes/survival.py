"""Governed cohort-level survival analysis routes."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse

from api.dependencies import AppState, get_app_state
from api.schemas.survival import CohortInsight, CoxModelResult, HazardOverTime, SurvivalAnalysisResponse
from src.serialization import json_safe

router = APIRouter(prefix='/api/survival', tags=['survival'])


def require_survival(state: AppState = Depends(get_app_state)) -> AppState:
    if not state.has_data():
        if not state.load_from_database():
            raise HTTPException(status_code=400, detail='No data loaded. Please upload a file first.')
    if state.survival_engine is None:
        raise HTTPException(status_code=409, detail='Cohort survival analysis is not available for the current dataset.')
    return state


def _json_response(model) -> JSONResponse:
    """Return a schema-validated model through the analytical JSON-safe boundary."""
    return JSONResponse(content=json_safe(model.model_dump(mode='python')))


@router.get('/analysis', response_model=SurvivalAnalysisResponse)
async def get_survival_analysis(state: AppState = Depends(require_survival)):
    results = json_safe(state.survival_engine.analyze_all())
    warnings = list(results.get('warnings', []))
    warnings.append('Individual employee survival ranking is disabled; this endpoint contains cohort-level retention evidence only.')
    response = SurvivalAnalysisResponse(
        kaplan_meier=results.get('kaplan_meier'),
        kaplan_meier_by_dept=results.get('kaplan_meier_by_dept'),
        cox_model=CoxModelResult(**results.get('cox_model', {})) if results.get('cox_model') else None,
        hazard_over_time=HazardOverTime(**results.get('hazard_over_time', {})) if results.get('hazard_over_time') else None,
        cohort_insights=[CohortInsight(**item) for item in results.get('cohort_insights', [])],
        at_risk_employees=[],
        summary=results.get('summary', {}),
        recommendations=results.get('recommendations', []),
        warnings=list(dict.fromkeys(warnings)),
    )
    return _json_response(response)


@router.get('/kaplan-meier')
async def get_kaplan_meier(
    segment_by: Optional[str] = Query(default=None, description='Column to segment by (for example Dept or Location)'),
    state: AppState = Depends(require_survival),
):
    return JSONResponse(content=json_safe(state.survival_engine.fit_kaplan_meier(segment_by=segment_by)))


@router.get('/cox-model', response_model=CoxModelResult)
async def get_cox_model(state: AppState = Depends(require_survival)):
    response = CoxModelResult(**json_safe(state.survival_engine.fit_cox_proportional_hazards()))
    return _json_response(response)


@router.get('/hazard-over-time', response_model=HazardOverTime)
async def get_hazard_over_time(state: AppState = Depends(require_survival)):
    response = HazardOverTime(**json_safe(state.survival_engine.get_hazard_over_time()))
    return _json_response(response)


@router.get('/cohort-insights', response_model=CohortInsight)
async def get_cohort_insights(
    dept: Optional[str] = Query(default=None),
    location: Optional[str] = Query(default=None),
    tenure_min: Optional[float] = Query(default=None),
    tenure_max: Optional[float] = Query(default=None),
    years_since_promotion_min: Optional[float] = Query(default=None),
    state: AppState = Depends(require_survival),
):
    filters = {}
    if dept:
        filters['Dept'] = dept
    if location:
        filters['Location'] = location
    if tenure_min is not None:
        filters['tenure_min'] = tenure_min
    if tenure_max is not None:
        filters['tenure_max'] = tenure_max
    if years_since_promotion_min is not None:
        filters['years_since_promotion_min'] = years_since_promotion_min
    response = CohortInsight(**json_safe(state.survival_engine.generate_cohort_insights(filters=filters or None)))
    return _json_response(response)


@router.get('/at-risk')
async def get_at_risk_employees(state: AppState = Depends(require_survival)):
    raise HTTPException(
        status_code=403,
        detail='Individual employee survival ranking is disabled. Use cohort retention evidence and governed investigation instead.',
    )


@router.get('/employee/{employee_id}')
async def get_employee_survival(employee_id: str, state: AppState = Depends(require_survival)):
    raise HTTPException(
        status_code=403,
        detail='Individual survival predictions are disabled. Survival analysis is available only at cohort level.',
    )
