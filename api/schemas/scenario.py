"""
Pydantic schemas for Scenario Planning API endpoints.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal, Union


class ScenarioTarget(BaseModel):
    """Target employee group for scenario."""
    scope: Literal['all', 'department', 'job_title', 'tenure_range', 'performance', 'custom'] = 'all'
    department: Optional[str] = Field(None, description="Department name when scope='department'")
    job_titles: Optional[List[str]] = Field(None, description="Job titles when scope='job_title'")
    tenure_min: Optional[float] = Field(None, description="Minimum tenure when scope='tenure_range'")
    tenure_max: Optional[float] = Field(None, description="Maximum tenure when scope='tenure_range'")
    performance_min: Optional[float] = Field(None, description="Minimum rating when scope='performance'")
    employee_ids: Optional[List[str]] = Field(None, description="Employee IDs when scope='custom'")


class CompensationScenarioRequest(BaseModel):
    """Request for compensation change scenario."""
    adjustment_type: Literal['percentage', 'absolute', 'market_adjustment'] = 'percentage'
    target: ScenarioTarget
    adjustment_value: float = Field(..., description="Adjustment amount (% or $ based on type)")
    time_horizon_months: int = Field(12, ge=1, le=60, description="Prediction horizon in months")


class HeadcountScenarioRequest(BaseModel):
    """Request for headcount change scenario."""
    change_type: Literal['reduction', 'expansion']
    target: ScenarioTarget
    change_count: Optional[int] = Field(None, ge=1, description="Absolute number to change")
    change_percentage: Optional[float] = Field(None, ge=0, le=100, description="Percentage to change")
    selection_criteria: Literal['performance', 'tenure', 'cost'] = 'performance'


class InterventionParams(BaseModel):
    """Parameters for retention intervention."""
    bonus_percentage: Optional[float] = Field(None, ge=0, le=100)
    cost_per_person: Optional[float] = Field(None, ge=0)
    change_cost: Optional[float] = Field(None, ge=0)


class InterventionScenarioRequest(BaseModel):
    """Request for attrition intervention scenario."""
    intervention_type: Literal['retention_bonus', 'career_path', 'manager_change']
    target_employees: Union[Literal['high_risk', 'high_risk_high_performer'], List[str]] = 'high_risk'
    intervention_params: InterventionParams = Field(default_factory=InterventionParams)


class CostImpactResponse(BaseModel):
    """Financial impact of a scenario."""
    salary_change: float = Field(0.0, description="Change in salary costs")
    replacement_costs_avoided: float = Field(0.0, description="Replacement costs saved")
    replacement_costs_incurred: float = Field(0.0, description="Replacement costs added")
    training_costs: float = Field(0.0, description="Training costs")
    productivity_impact: float = Field(0.0, description="Productivity impact in $")
    total_cost: float = Field(0.0, description="Total cost of scenario")
    total_benefit: float = Field(0.0, description="Total benefit of scenario")
    net_impact: float = Field(0.0, description="Net financial impact")


class MonteCarloResultResponse(BaseModel):
    """Results from Monte Carlo simulation."""
    n_iterations: int = Field(..., description="Number of simulation iterations")
    outcome_mean: float = Field(..., description="Mean outcome (e.g., turnover rate)")
    outcome_std: float = Field(..., description="Standard deviation of outcome")
    outcome_median: float = Field(..., description="Median outcome")
    percentiles: Dict[str, float] = Field(..., description="Percentile values (p5, p25, p50, p75, p95)")
    histogram_bins: List[float] = Field(..., description="Histogram bin edges")
    histogram_counts: List[int] = Field(..., description="Histogram bin counts")
    cost_impact_mean: float = Field(..., description="Mean cost impact")
    cost_impact_std: float = Field(..., description="Std dev of cost impact")
    cost_impact_percentiles: Dict[str, float] = Field(..., description="Cost impact percentiles")
    roi_mean: float = Field(..., description="Mean ROI percentage")
    roi_std: float = Field(..., description="Std dev of ROI")
    roi_positive_probability: float = Field(..., description="Probability of positive ROI (0-1)")
    converged: bool = Field(..., description="Whether simulation converged")
    convergence_iterations: int = Field(..., description="Iterations to convergence")


class ScenarioResultResponse(BaseModel):
    """Complete scenario simulation result."""
    available: bool = True
    scenario_id: str = Field(..., description="Unique scenario identifier")
    scenario_name: str = Field(..., description="Human-readable scenario name")
    scenario_type: str = Field(..., description="compensation, headcount, or intervention")
    input_parameters: Dict[str, Any] = Field(..., description="Input parameters used")

    affected_employees: int = Field(..., description="Number of employees affected")
    affected_departments: List[str] = Field(default_factory=list, description="Departments affected")

    baseline_turnover_rate: float = Field(..., description="Baseline turnover rate (%)")
    projected_turnover_rate: float = Field(..., description="Projected turnover rate (%)")
    turnover_change: float = Field(..., description="Absolute turnover change (pp)")
    turnover_change_pct: float = Field(..., description="Relative turnover change (%)")

    simulation: MonteCarloResultResponse = Field(..., description="Monte Carlo simulation results")
    cost_impact: CostImpactResponse = Field(..., description="Financial impact analysis")

    roi_estimate: Optional[float] = Field(None, description="Expected ROI percentage")
    payback_months: Optional[int] = Field(None, description="Months to payback")

    confidence_level: str = Field(..., description="High, Medium, or Low")
    confidence_score: float = Field(..., description="Confidence score (0-1)")
    assumptions: List[str] = Field(default_factory=list, description="Key assumptions")
    risks: List[str] = Field(default_factory=list, description="Identified risks")
    recommendation: str = Field(..., description="Overall recommendation")
    alternative_actions: List[str] = Field(default_factory=list, description="Alternative approaches")

    computed_at: str = Field(..., description="ISO timestamp of computation")
    engines_used: List[str] = Field(default_factory=list, description="Engines used for prediction")
    data_sources: List[str] = Field(default_factory=list, description="Data sources used for predictions")


class ScenarioComparisonRequest(BaseModel):
    """Request to compare multiple scenarios."""
    scenario_ids: List[str] = Field(..., min_length=2, description="Scenario IDs to compare")


class ScenarioComparisonItem(BaseModel):
    """Summary of a scenario for comparison."""
    scenario_id: str
    scenario_name: str
    affected_employees: int
    turnover_change_pct: float
    roi_estimate: Optional[float]
    net_impact: float
    confidence_level: str
    roi_positive_probability: float


class ScenarioComparisonResponse(BaseModel):
    """Response comparing multiple scenarios."""
    available: bool = True
    scenarios: List[ScenarioComparisonItem]
    recommended_scenario: str = Field(..., description="Name of recommended scenario")
    reasoning: str = Field(..., description="Explanation for recommendation")


class ScenarioTemplate(BaseModel):
    """Pre-built scenario template."""
    name: str
    type: str
    description: str = ""
    params: Dict[str, Any] = Field(default_factory=dict)


class ScenarioTemplatesResponse(BaseModel):
    """Response listing available templates."""
    available: bool = True
    templates: List[ScenarioTemplate]


class SensitivityRequest(BaseModel):
    """Request for sensitivity analysis."""
    scenario_type: Literal['compensation', 'headcount', 'intervention']
    base_request: Dict[str, Any] = Field(..., description="Base scenario parameters")
    variable: str = Field(..., description="Variable to vary (e.g., 'adjustment_value')")
    range_values: List[float] = Field(..., description="Values to test")


class SensitivityPoint(BaseModel):
    """Single point in sensitivity analysis."""
    variable_value: float
    roi_estimate: float
    turnover_change_pct: float
    net_impact: float


class SensitivityResponse(BaseModel):
    """Response for sensitivity analysis."""
    available: bool = True
    variable: str
    points: List[SensitivityPoint]
    optimal_value: float = Field(..., description="Value with best outcome")
    insight: str = Field(..., description="Key insight from analysis")


class ScenarioErrorResponse(BaseModel):
    """Error response for scenario endpoints."""
    available: bool = False
    error: str
    detail: Optional[str] = None
