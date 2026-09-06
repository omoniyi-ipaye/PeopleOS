"""Governed aggregate structural-analysis routes.

PeopleOS surfaces department/system structure signals. Individual stagnation,
manager ranking and employee waiting lists are not exposed because those labels
can influence consequential people decisions without sufficient context.
"""

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import AppState, get_app_state

router = APIRouter(prefix="/api/structural", tags=["structural"])


def require_structural(state: AppState = Depends(get_app_state)) -> AppState:
    if not state.has_data():
        if not state.load_from_database():
            raise HTTPException(status_code=400, detail="No data loaded. Please upload a file first.")
    if state.structural_engine is None:
        raise HTTPException(status_code=400, detail="Structural analysis is unavailable for the current dataset.")
    return state


def _blocked(detail: str) -> HTTPException:
    return HTTPException(status_code=403, detail=detail)


@router.get("/analysis")
async def get_structural_analysis(state: AppState = Depends(require_structural)):
    engine = state.structural_engine
    stagnation = engine.identify_stagnation_hotspots()
    span = engine.analyze_manager_burnout_risk()
    promotion = engine.audit_promotion_velocity()
    bottlenecks = engine.get_promotion_bottlenecks()

    if isinstance(stagnation, dict):
        stagnation = {k: v for k, v in stagnation.items() if k not in {'critical_employees', 'employees'}}
    if isinstance(span, dict):
        span = {k: v for k, v in span.items() if k not in {'at_risk_managers', 'managers'}}
    if isinstance(bottlenecks, dict):
        bottlenecks = {k: v for k, v in bottlenecks.items() if k not in {'employees_waiting_longest', 'employees'}}
    if isinstance(promotion, dict):
        promotion['recommendations'] = ["Treat group differences as screening evidence and review methodology, sample size and context before policy changes."]

    return {
        'stagnation': stagnation,
        'span_of_control': span,
        'promotion_equity': promotion,
        'promotion_bottlenecks': bottlenecks,
        'governance': 'Aggregate structural screening only; no employee or manager ranking is exposed.',
    }


@router.get("/stagnation", deprecated=True)
async def get_stagnation_index(state: AppState = Depends(require_structural)):
    raise _blocked("Individual stagnation ranking is disabled. Use aggregate structural hotspots.")


@router.get("/stagnation/hotspots")
async def get_stagnation_hotspots(state: AppState = Depends(require_structural)):
    result = state.structural_engine.identify_stagnation_hotspots()
    if not isinstance(result, dict):
        return result
    return {k: v for k, v in result.items() if k not in {'critical_employees', 'employees'}} | {
        'metric_semantics': 'aggregate_role_tenure_screening_not_employee_performance_determination'
    }


@router.get("/span-of-control", deprecated=True)
async def get_span_of_control(state: AppState = Depends(require_structural)):
    raise _blocked("Named manager span ranking is disabled. Use aggregate department span analysis.")


@router.get("/span-of-control/analysis")
async def get_span_analysis(state: AppState = Depends(require_structural)):
    result = state.structural_engine.analyze_manager_burnout_risk()
    if not isinstance(result, dict):
        return result
    safe = {k: v for k, v in result.items() if k not in {'at_risk_managers', 'managers'}}
    safe['recommendations'] = ["Use span thresholds as structural workload prompts, not a diagnosis of manager burnout."]
    safe['metric_semantics'] = 'span_of_control_screening_not_burnout_diagnosis'
    return safe


@router.get("/promotion-equity")
async def get_promotion_equity_audit(state: AppState = Depends(require_structural)):
    result = state.structural_engine.audit_promotion_velocity()
    if isinstance(result, dict) and result.get('available'):
        result['recommendations'] = ["Treat observed group differences as equity-screening evidence; validate role mix, tenure, level and sample-size effects before action."]
        result['metric_semantics'] = 'observational_promotion_velocity_screening_not_causal_discrimination_finding'
    return result


@router.get("/promotion-bottlenecks")
async def get_promotion_bottlenecks(state: AppState = Depends(require_structural)):
    result = state.structural_engine.get_promotion_bottlenecks()
    if not isinstance(result, dict):
        return result
    safe = {k: v for k, v in result.items() if k not in {'employees_waiting_longest', 'employees'}}
    safe['metric_semantics'] = 'aggregate_wait_time_comparison_not_individual_promotion_recommendation'
    return safe


@router.get("/employee/{employee_id}/stagnation", deprecated=True)
async def get_employee_stagnation(employee_id: str, state: AppState = Depends(require_structural)):
    raise _blocked("Individual stagnation scoring is disabled. Use aggregate structural analysis.")
