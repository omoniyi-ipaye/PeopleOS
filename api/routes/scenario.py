"""
API routes for Scenario Planning endpoints.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any

from api.dependencies import get_app_state, AppState
from api.schemas.scenario import (
    CompensationScenarioRequest,
    HeadcountScenarioRequest,
    InterventionScenarioRequest,
    ScenarioResultResponse,
    ScenarioComparisonResponse,
    ScenarioComparisonItem,
    ScenarioTemplatesResponse,
    ScenarioTemplate,
    SensitivityRequest,
    SensitivityResponse,
    SensitivityPoint,
    CostImpactResponse,
    MonteCarloResultResponse,
)
from src.scenario_engine import ScenarioEngineError

router = APIRouter(prefix="/api/scenario", tags=["scenario"])

# Store recent scenarios for comparison (in-memory for simplicity)
_scenario_cache: Dict[str, Dict[str, Any]] = {}


def require_scenario(state: AppState = Depends(get_app_state)) -> AppState:
    """Dependency that requires scenario engine to be available."""
    if not state.has_data():
        if not state.load_from_database():
            raise HTTPException(
                status_code=400,
                detail="No data loaded. Please upload a file first."
            )

    if state.scenario_engine is None:
        raise HTTPException(
            status_code=400,
            detail="Scenario planning not available. Engine initialization may have failed."
        )

    return state


def _convert_result(result) -> ScenarioResultResponse:
    """Convert ScenarioResult dataclass to response model."""
    from dataclasses import asdict

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
        convergence_iterations=result.simulation.convergence_iterations
    )

    cost_impact = CostImpactResponse(
        salary_change=result.cost_impact.salary_change,
        replacement_costs_avoided=result.cost_impact.replacement_costs_avoided,
        replacement_costs_incurred=result.cost_impact.replacement_costs_incurred,
        training_costs=result.cost_impact.training_costs,
        productivity_impact=result.cost_impact.productivity_impact,
        total_cost=result.cost_impact.total_cost,
        total_benefit=result.cost_impact.total_benefit,
        net_impact=result.cost_impact.net_impact
    )

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
        engines_used=result.engines_used
    )

    # Cache for comparison
    _scenario_cache[result.scenario_id] = asdict(result)

    return response


@router.post("/simulate/compensation", response_model=ScenarioResultResponse)
async def simulate_compensation_change(
    request: CompensationScenarioRequest,
    state: AppState = Depends(require_scenario)
) -> ScenarioResultResponse:
    """
    Simulate the impact of compensation changes on turnover.

    Models:
    - Percentage raises (e.g., 5% raise to Engineering)
    - Absolute adjustments (e.g., $5000 increase)
    - Market adjustments

    Returns predicted turnover impact, ROI, and Monte Carlo distribution.
    """
    try:
        result = state.scenario_engine.simulate_compensation_change(
            adjustment_type=request.adjustment_type,
            target=request.target.model_dump(),
            adjustment_value=request.adjustment_value,
            time_horizon_months=request.time_horizon_months
        )
        return _convert_result(result)
    except ScenarioEngineError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/simulate/headcount", response_model=ScenarioResultResponse)
async def simulate_headcount_change(
    request: HeadcountScenarioRequest,
    state: AppState = Depends(require_scenario)
) -> ScenarioResultResponse:
    """
    Simulate the impact of headcount changes.

    Models:
    - Workforce reduction (layoffs, attrition)
    - Workforce expansion (hiring)

    Returns cost impact, productivity effects, and risk assessment.
    """
    try:
        result = state.scenario_engine.simulate_headcount_change(
            change_type=request.change_type,
            target=request.target.model_dump(),
            change_count=request.change_count,
            change_percentage=request.change_percentage,
            selection_criteria=request.selection_criteria
        )
        return _convert_result(result)
    except ScenarioEngineError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/simulate/intervention", response_model=ScenarioResultResponse)
async def simulate_attrition_intervention(
    request: InterventionScenarioRequest,
    state: AppState = Depends(require_scenario)
) -> ScenarioResultResponse:
    """
    Simulate the effectiveness of retention interventions.

    Intervention types:
    - retention_bonus: Cash bonus to retain employees
    - career_path: Development/promotion program
    - manager_change: Organizational restructuring

    Returns expected retention improvement and ROI.
    """
    try:
        target = request.target_employees
        if isinstance(target, list):
            pass  # Already a list of IDs
        # Otherwise it's a string like 'high_risk'

        result = state.scenario_engine.simulate_attrition_intervention(
            intervention_type=request.intervention_type,
            target_employees=target,
            intervention_params=request.intervention_params.model_dump()
        )
        return _convert_result(result)
    except ScenarioEngineError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/templates", response_model=ScenarioTemplatesResponse)
async def get_scenario_templates(
    state: AppState = Depends(require_scenario)
) -> ScenarioTemplatesResponse:
    """
    Get pre-built scenario templates.

    Templates provide common scenario configurations that can be
    used as starting points for analysis.
    """
    templates = state.scenario_engine.get_scenario_templates()
    return ScenarioTemplatesResponse(
        templates=[ScenarioTemplate(**t) for t in templates]
    )


@router.post("/compare", response_model=ScenarioComparisonResponse)
async def compare_scenarios(
    scenario_ids: list[str],
    state: AppState = Depends(require_scenario)
) -> ScenarioComparisonResponse:
    """
    Compare multiple scenario results side-by-side.

    Requires at least 2 previously run scenarios (by ID).
    Returns ranked comparison with recommendation.
    """
    # Retrieve cached scenarios
    scenarios = []
    for sid in scenario_ids:
        if sid not in _scenario_cache:
            raise HTTPException(
                status_code=404,
                detail=f"Scenario {sid} not found. Run the scenario first."
            )
        scenarios.append(_scenario_cache[sid])

    if len(scenarios) < 2:
        raise HTTPException(
            status_code=400,
            detail="Need at least 2 scenarios to compare"
        )

    # Build comparison
    comparison_items = []
    for s in scenarios:
        comparison_items.append(ScenarioComparisonItem(
            scenario_id=s['scenario_id'],
            scenario_name=s['scenario_name'],
            affected_employees=s['affected_employees'],
            turnover_change_pct=s['turnover_change_pct'],
            roi_estimate=s['roi_estimate'],
            net_impact=s['cost_impact']['net_impact'],
            confidence_level=s['confidence_level'],
            roi_positive_probability=s['simulation']['roi_positive_probability']
        ))

    # Sort by ROI
    comparison_items.sort(key=lambda x: x.roi_estimate or 0, reverse=True)
    best = comparison_items[0]

    return ScenarioComparisonResponse(
        scenarios=comparison_items,
        recommended_scenario=best.scenario_name,
        reasoning=f"Highest expected ROI ({best.roi_estimate}%) with "
                  f"{best.roi_positive_probability*100:.0f}% probability of positive return"
    )


@router.get("/{scenario_id}", response_model=ScenarioResultResponse)
async def get_scenario(
    scenario_id: str,
    state: AppState = Depends(require_scenario)
) -> ScenarioResultResponse:
    """
    Retrieve a previously run scenario by ID.
    """
    if scenario_id not in _scenario_cache:
        raise HTTPException(
            status_code=404,
            detail=f"Scenario {scenario_id} not found"
        )

    s = _scenario_cache[scenario_id]

    # Reconstruct response from cache
    simulation = MonteCarloResultResponse(**s['simulation'])
    cost_impact = CostImpactResponse(**s['cost_impact'])

    return ScenarioResultResponse(
        scenario_id=s['scenario_id'],
        scenario_name=s['scenario_name'],
        scenario_type=s['scenario_type'],
        input_parameters=s['input_parameters'],
        affected_employees=s['affected_employees'],
        affected_departments=s['affected_departments'],
        baseline_turnover_rate=s['baseline_turnover_rate'],
        projected_turnover_rate=s['projected_turnover_rate'],
        turnover_change=s['turnover_change'],
        turnover_change_pct=s['turnover_change_pct'],
        simulation=simulation,
        cost_impact=cost_impact,
        roi_estimate=s['roi_estimate'],
        payback_months=s['payback_months'],
        confidence_level=s['confidence_level'],
        confidence_score=s['confidence_score'],
        assumptions=s['assumptions'],
        risks=s['risks'],
        recommendation=s['recommendation'],
        alternative_actions=s['alternative_actions'],
        computed_at=s['computed_at'],
        engines_used=s['engines_used']
    )


@router.delete("/{scenario_id}")
async def delete_scenario(
    scenario_id: str,
    state: AppState = Depends(require_scenario)
):
    """
    Delete a saved scenario from cache.
    """
    if scenario_id not in _scenario_cache:
        raise HTTPException(
            status_code=404,
            detail=f"Scenario {scenario_id} not found"
        )

    del _scenario_cache[scenario_id]
    return {"deleted": True, "scenario_id": scenario_id}


@router.post("/sensitivity", response_model=SensitivityResponse)
async def analyze_sensitivity(
    request: SensitivityRequest,
    state: AppState = Depends(require_scenario)
) -> SensitivityResponse:
    """
    Analyze how sensitive outcomes are to changes in one variable.

    Runs multiple scenarios varying the specified parameter
    to identify optimal values and break-even points.
    """
    points = []

    for value in request.range_values:
        try:
            # Modify the base request with new value
            modified_request = request.base_request.copy()
            modified_request[request.variable] = value

            # Run appropriate scenario type
            if request.scenario_type == 'compensation':
                result = state.scenario_engine.simulate_compensation_change(
                    adjustment_type=modified_request.get('adjustment_type', 'percentage'),
                    target=modified_request.get('target', {'scope': 'all'}),
                    adjustment_value=modified_request.get('adjustment_value', value),
                    time_horizon_months=modified_request.get('time_horizon_months', 12)
                )
            elif request.scenario_type == 'headcount':
                result = state.scenario_engine.simulate_headcount_change(
                    change_type=modified_request.get('change_type', 'reduction'),
                    target=modified_request.get('target', {'scope': 'all'}),
                    change_count=modified_request.get('change_count'),
                    change_percentage=modified_request.get('change_percentage'),
                    selection_criteria=modified_request.get('selection_criteria', 'performance')
                )
            else:  # intervention
                result = state.scenario_engine.simulate_attrition_intervention(
                    intervention_type=modified_request.get('intervention_type', 'retention_bonus'),
                    target_employees=modified_request.get('target_employees', 'high_risk'),
                    intervention_params=modified_request.get('intervention_params', {})
                )

            points.append(SensitivityPoint(
                variable_value=value,
                roi_estimate=result.roi_estimate or 0,
                turnover_change_pct=result.turnover_change_pct,
                net_impact=result.cost_impact.net_impact
            ))

        except ScenarioEngineError:
            continue

    if not points:
        raise HTTPException(
            status_code=400,
            detail="Could not generate sensitivity analysis points"
        )

    # Find optimal value (max ROI)
    optimal = max(points, key=lambda p: p.roi_estimate)

    # Generate insight
    if optimal.roi_estimate > 50:
        insight = f"Strong ROI at {request.variable}={optimal.variable_value}. Consider implementing."
    elif optimal.roi_estimate > 0:
        insight = f"Positive ROI achievable at {request.variable}={optimal.variable_value}."
    else:
        insight = f"No positive ROI found in tested range. Consider alternative approaches."

    return SensitivityResponse(
        variable=request.variable,
        points=points,
        optimal_value=optimal.variable_value,
        insight=insight
    )


@router.get("/history/recent")
async def get_recent_scenarios(
    limit: int = 10,
    state: AppState = Depends(require_scenario)
):
    """
    Get recently run scenarios.
    """
    scenarios = list(_scenario_cache.values())[-limit:]
    return {
        "available": True,
        "count": len(scenarios),
        "scenarios": [
            {
                "scenario_id": s['scenario_id'],
                "scenario_name": s['scenario_name'],
                "scenario_type": s['scenario_type'],
                "computed_at": s['computed_at'],
                "roi_estimate": s['roi_estimate']
            }
            for s in scenarios
        ]
    }
