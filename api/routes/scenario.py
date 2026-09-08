"""Governed Scenario Planning API routes.

Scenario Planner is an exploratory decision-support surface. Historical
associations, configured cost multipliers and Monte Carlo draws are not causal
forecasts. This API boundary prevents legacy scenario heuristics from being
presented as validated predictions or employee-selection recommendations.
"""

from dataclasses import asdict
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Query
from src.platform.provenance import IntegrityError, snapshot_provenance

from api.dependencies import AppState, get_app_state
from api.schemas.scenario import (
    CompensationScenarioRequest,
    CostImpactResponse,
    HeadcountScenarioRequest,
    InterventionScenarioRequest,
    MonteCarloResultResponse,
    ScenarioComparisonItem,
    ScenarioComparisonResponse,
    ScenarioResultResponse,
    ScenarioTemplate,
    ScenarioTemplatesResponse,
    SensitivityPoint,
    SensitivityRequest,
    SensitivityResponse,
)
from src.scenario_engine import ScenarioEngineError

router = APIRouter(prefix="/api/scenario", tags=["scenario"])
def _cache(state):
    try:
        snapshot_provenance(state)
    except IntegrityError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if not hasattr(state, 'scenario_cache'):
        state.scenario_cache = {}
    return state.scenario_cache


def _cached_scenario(state, scenario_id):
    item = _cache(state).get(scenario_id)
    if item is None:
        raise HTTPException(status_code=404, detail='Scenario is unavailable in the active dataset snapshot. Run it again.')
    if item.get('provenance') != snapshot_provenance(state):
        raise HTTPException(status_code=409, detail='Scenario belongs to another dataset snapshot.')
    return item

_INTEGRITY_ASSUMPTIONS = [
    "Exploratory scenario only: modeled relationships are associative and assumption-based, not causal treatment-effect estimates.",
    "Monte Carlo run shares reflect the configured input distribution; they are not empirical probabilities of future outcomes.",
    "Replacement cost, productivity, intervention and pay-response parameters require local finance/People validation before use.",
]


def require_scenario(state: AppState = Depends(get_app_state)) -> AppState:
    if not state.has_data():
        if not state.load_from_database():
            raise HTTPException(status_code=400, detail="No data loaded. Please upload a file first.")
    if state.scenario_engine is None:
        raise HTTPException(status_code=400, detail="Scenario planning is unavailable for the current dataset.")
    return state


def _sanitize_result(result):
    """Apply product-level integrity semantics to a legacy scenario result."""
    assumptions = list(result.assumptions or [])
    for assumption in _INTEGRITY_ASSUMPTIONS:
        if assumption not in assumptions:
            assumptions.append(assumption)
    result.assumptions = assumptions
    result.confidence_level = "Exploratory"
    result.confidence_score = min(float(result.confidence_score or 0.0), 0.5)
    result.recommendation = (
        "Compare this modeled scenario with alternatives and validate the causal and financial assumptions before making a workforce decision."
    )
    result.alternative_actions = [
        action for action in (result.alternative_actions or [])
        if "high-risk" not in action.lower() and "employee" not in action.lower()
    ]
    if "Run a smaller pilot or prospective test before scaling" not in result.alternative_actions:
        result.alternative_actions.append("Run a smaller pilot or prospective test before scaling")
    return result


def _convert_result(result, state=None) -> ScenarioResultResponse:
    result = _sanitize_result(result)
    simulation = MonteCarloResultResponse(
        n_iterations=result.simulation.n_iterations,
        outcome_mean=result.simulation.outcome_mean,
        outcome_std=result.simulation.outcome_std,
        outcome_median=result.simulation.outcome_median,
        percentiles=result.simulation.percentiles,
        histogram_bins=result.simulation.histogram_bins,
        histogram_counts=result.simulation.histogram_counts,
        cost_impact_mean=result.simulation.cost_impact_mean,
        cost_impact_std=result.simulation.cost_impact_std,
        cost_impact_percentiles=result.simulation.cost_impact_percentiles,
        roi_mean=result.simulation.roi_mean,
        roi_std=result.simulation.roi_std,
        roi_positive_probability=result.simulation.roi_positive_probability,
        converged=result.simulation.converged,
        convergence_iterations=result.simulation.convergence_iterations,
    )
    cost_impact = CostImpactResponse(**asdict(result.cost_impact))
    response = ScenarioResultResponse(
        scenario_id=result.scenario_id,
        scenario_name=result.scenario_name,
        scenario_type=result.scenario_type,
        input_parameters=result.input_parameters,
        affected_employees=result.affected_employees,
        affected_departments=result.affected_departments,
        baseline_turnover_rate=result.baseline_turnover_rate,
        projected_turnover_rate=result.projected_turnover_rate,
        turnover_change=result.turnover_change,
        turnover_change_pct=result.turnover_change_pct,
        simulation=simulation,
        cost_impact=cost_impact,
        roi_estimate=result.roi_estimate,
        payback_months=result.payback_months,
        confidence_level=result.confidence_level,
        confidence_score=result.confidence_score,
        assumptions=result.assumptions,
        risks=result.risks,
        recommendation=result.recommendation,
        alternative_actions=result.alternative_actions,
        computed_at=result.computed_at,
        engines_used=result.engines_used,
        data_sources=result.data_sources,
    )
    if state is not None:
        provenance = snapshot_provenance(state)
        response.provenance = provenance
        _cache(state)[result.scenario_id] = {**asdict(result), 'provenance': provenance}
        # Bounded local session history; activation clears it.
        while len(state.scenario_cache) > 100:
            del state.scenario_cache[next(iter(state.scenario_cache))]
    return response


@router.post("/simulate/compensation", response_model=ScenarioResultResponse)
async def simulate_compensation_change(
    request: CompensationScenarioRequest,
    state: AppState = Depends(require_scenario),
) -> ScenarioResultResponse:
    """Explore a compensation scenario under explicit assumptions."""
    try:
        result = state.scenario_engine.simulate_compensation_change(
            adjustment_type=request.adjustment_type,
            target=request.target.model_dump(),
            adjustment_value=request.adjustment_value,
            time_horizon_months=request.time_horizon_months,
        )
        return _convert_result(result, state)
    except ScenarioEngineError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/simulate/headcount", response_model=ScenarioResultResponse)
async def simulate_headcount_change(
    request: HeadcountScenarioRequest,
    state: AppState = Depends(require_scenario),
) -> ScenarioResultResponse:
    """Explore aggregate expansion only; employee-selection reductions are disabled."""
    if request.change_type == "reduction":
        raise HTTPException(
            status_code=409,
            detail=(
                "Headcount-reduction simulation is disabled because the legacy model selects individuals by performance, tenure, or cost. "
                "PeopleOS will not rank employees for consequential workforce actions."
            ),
        )
    try:
        result = state.scenario_engine.simulate_headcount_change(
            change_type="expansion",
            target=request.target.model_dump(),
            change_count=request.change_count,
            change_percentage=request.change_percentage,
            selection_criteria="performance",
        )
        return _convert_result(result, state)
    except ScenarioEngineError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/simulate/intervention", response_model=ScenarioResultResponse, deprecated=True)
async def simulate_attrition_intervention(
    request: InterventionScenarioRequest,
    state: AppState = Depends(require_scenario),
) -> ScenarioResultResponse:
    """Disabled until a validated aggregate intervention-effect model exists."""
    raise HTTPException(
        status_code=409,
        detail=(
            "Retention-intervention simulation is temporarily disabled. The legacy implementation targets employee risk groups and uses unvalidated intervention-effect assumptions."
        ),
    )


@router.get("/templates", response_model=ScenarioTemplatesResponse)
async def get_scenario_templates(state: AppState = Depends(require_scenario)) -> ScenarioTemplatesResponse:
    templates = state.scenario_engine.get_scenario_templates()
    # Exclude templates that depend on individual-risk targeting or reduction selection.
    safe_templates = [
        item for item in templates
        if item.get("type") == "compensation" or (item.get("type") == "headcount" and item.get("params", {}).get("change_type") != "reduction")
    ]
    return ScenarioTemplatesResponse(templates=[ScenarioTemplate(**item) for item in safe_templates])


@router.post("/compare", response_model=ScenarioComparisonResponse)
async def compare_scenarios(
    scenario_ids: list[str],
    state: AppState = Depends(require_scenario),
) -> ScenarioComparisonResponse:
    scenarios = []
    for scenario_id in scenario_ids:
        scenarios.append(_cached_scenario(state, scenario_id))
    if len(scenarios) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 scenarios to compare")

    horizons = {item['input_parameters'].get('time_horizon_months', 12) for item in scenarios}
    if len(horizons) != 1:
        raise HTTPException(status_code=409, detail='Scenarios must use the same time horizon for comparison.')
    items = [
        ScenarioComparisonItem(
            scenario_id=item["scenario_id"],
            scenario_name=item["scenario_name"],
            affected_employees=item["affected_employees"],
            turnover_change_pct=item["turnover_change_pct"],
            roi_estimate=item["roi_estimate"],
            net_impact=item["cost_impact"]["net_impact"],
            confidence_level="Exploratory",
            roi_positive_probability=item["simulation"]["roi_positive_probability"],
        )
        for item in scenarios
    ]
    return ScenarioComparisonResponse(
        scenarios=items,
        recommended_scenario="No automatic recommendation",
        reasoning=(
            "Scenario outputs depend on unvalidated causal and financial assumptions. Compare sensitivity, feasibility and downside risk with accountable People/Finance owners rather than selecting by modeled ROI alone."
        ),
    )


@router.get("/{scenario_id}", response_model=ScenarioResultResponse)
async def get_scenario(scenario_id: str, state: AppState = Depends(require_scenario)) -> ScenarioResultResponse:
    item = _cached_scenario(state, scenario_id)
    return ScenarioResultResponse(
        provenance=item["provenance"],
        scenario_id=item["scenario_id"],
        scenario_name=item["scenario_name"],
        scenario_type=item["scenario_type"],
        input_parameters=item["input_parameters"],
        affected_employees=item["affected_employees"],
        affected_departments=item["affected_departments"],
        baseline_turnover_rate=item["baseline_turnover_rate"],
        projected_turnover_rate=item["projected_turnover_rate"],
        turnover_change=item["turnover_change"],
        turnover_change_pct=item["turnover_change_pct"],
        simulation=MonteCarloResultResponse(**item["simulation"]),
        cost_impact=CostImpactResponse(**item["cost_impact"]),
        roi_estimate=item["roi_estimate"],
        payback_months=item["payback_months"],
        confidence_level="Exploratory",
        confidence_score=min(float(item.get("confidence_score", 0.0)), 0.5),
        assumptions=item.get("assumptions", []) + [a for a in _INTEGRITY_ASSUMPTIONS if a not in item.get("assumptions", [])],
        risks=item.get("risks", []),
        recommendation="Compare this scenario with alternatives and validate assumptions before action.",
        alternative_actions=item.get("alternative_actions", []),
        computed_at=item["computed_at"],
        engines_used=item.get("engines_used", []),
        data_sources=item.get("data_sources", []),
    )


@router.delete("/{scenario_id}")
async def delete_scenario(scenario_id: str, state: AppState = Depends(require_scenario)):
    _cached_scenario(state, scenario_id)
    del state.scenario_cache[scenario_id]
    return {"deleted": True, "scenario_id": scenario_id}


@router.post("/sensitivity", response_model=SensitivityResponse)
async def analyze_sensitivity(
    request: SensitivityRequest,
    state: AppState = Depends(require_scenario),
) -> SensitivityResponse:
    if request.scenario_type == "intervention":
        raise HTTPException(status_code=409, detail="Intervention sensitivity is disabled pending a validated aggregate effect model.")
    if request.scenario_type == "headcount" and request.base_request.get("change_type") == "reduction":
        raise HTTPException(status_code=409, detail="Reduction sensitivity is disabled because individual selection is outside PeopleOS autonomy.")

    points: list[SensitivityPoint] = []
    for value in request.range_values:
        try:
            modified = request.base_request.copy()
            modified[request.variable] = value
            if request.scenario_type == "compensation":
                result = state.scenario_engine.simulate_compensation_change(
                    adjustment_type=modified.get("adjustment_type", "percentage"),
                    target=modified.get("target", {"scope": "all"}),
                    adjustment_value=modified.get("adjustment_value", value),
                    time_horizon_months=modified.get("time_horizon_months", 12),
                )
            else:
                result = state.scenario_engine.simulate_headcount_change(
                    change_type="expansion",
                    target=modified.get("target", {"scope": "all"}),
                    change_count=modified.get("change_count"),
                    change_percentage=modified.get("change_percentage"),
                    selection_criteria="performance",
                )
            points.append(SensitivityPoint(
                variable_value=value,
                roi_estimate=result.roi_estimate or 0,
                turnover_change_pct=result.turnover_change_pct,
                net_impact=result.cost_impact.net_impact,
            ))
        except ScenarioEngineError:
            continue

    if not points:
        raise HTTPException(status_code=400, detail="Could not generate sensitivity analysis points")
    best_modeled = max(points, key=lambda point: point.roi_estimate)
    return SensitivityResponse(
        variable=request.variable,
        points=points,
        optimal_value=best_modeled.variable_value,
        insight=(
            f"Under the current assumptions, {request.variable}={best_modeled.variable_value} produces the highest modeled ROI in the tested range. "
            "This is an assumption-sensitivity result, not an implementation recommendation."
        ),
    )


@router.get("/history/recent")
async def get_recent_scenarios(limit: int = Query(default=10, ge=1, le=100), state: AppState = Depends(require_scenario)):
    scenarios = list(_cache(state).values())[-limit:]
    return {
        "available": True,
        "count": len(scenarios),
        "scenarios": [
            {
                "scenario_id": item["scenario_id"],
                "scenario_name": item["scenario_name"],
                "scenario_type": item["scenario_type"],
                "computed_at": item["computed_at"],
                "roi_estimate": item["roi_estimate"],
                "evidence_strength": "exploratory",
                "provenance": item["provenance"],
            }
            for item in scenarios
        ],
    }
