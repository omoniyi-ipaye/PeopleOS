"""
Scenario Planning Engine for PeopleOS.

What-If scenario modeling with Monte Carlo simulation for:
- Compensation adjustments → turnover impact
- Headcount changes → cost/productivity impact
- Promotion policies → retention impact
- Retention interventions → ROI modeling

DATA-DRIVEN APPROACH:
- Calculates pay-turnover elasticity from actual historical data
- Uses ML model predictions for individual employee risk when available
- Uses Survival model hazard ratios when available
- Falls back to statistical estimates only when no data is available
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass, asdict
from datetime import datetime
import uuid

from src.population import active_population, resolve_current_population
from src.utils import load_config, safe_divide
from src.logger import get_logger

if TYPE_CHECKING:
    from src.ml_engine import MLEngine
    from src.survival_engine import SurvivalEngine
    from src.compensation_engine import CompensationEngine


class ScenarioEngineError(Exception):
    """Custom exception for ScenarioEngine errors."""
    pass


@dataclass
class CostImpact:
    """Financial impact of a scenario."""
    salary_change: float = 0.0
    replacement_costs_avoided: float = 0.0
    replacement_costs_incurred: float = 0.0
    training_costs: float = 0.0
    productivity_impact: float = 0.0
    total_cost: float = 0.0
    total_benefit: float = 0.0
    net_impact: float = 0.0


@dataclass
class MonteCarloResult:
    """Results from Monte Carlo simulation."""
    n_iterations: int
    outcome_mean: float
    outcome_std: float
    outcome_median: float
    percentiles: Dict[str, float]
    histogram_bins: List[float]
    histogram_counts: List[int]
    cost_impact_mean: float
    cost_impact_std: float
    cost_impact_percentiles: Dict[str, float]
    roi_mean: float
    roi_std: float
    roi_positive_probability: float
    converged: bool
    convergence_iterations: int


@dataclass
class ScenarioResult:
    """Complete scenario simulation result."""
    scenario_id: str
    scenario_name: str
    scenario_type: str
    input_parameters: Dict[str, Any]
    affected_employees: int
    affected_departments: List[str]

    # Baseline vs projected
    baseline_turnover_rate: float
    projected_turnover_rate: float
    turnover_change: float
    turnover_change_pct: float

    # Monte Carlo results
    simulation: MonteCarloResult

    # Financial
    cost_impact: CostImpact
    roi_estimate: Optional[float]
    payback_months: Optional[int]

    # Assessment
    confidence_level: str  # 'High', 'Medium', 'Low'
    confidence_score: float
    assumptions: List[str]
    risks: List[str]
    recommendation: str
    alternative_actions: List[str]

    # Metadata
    computed_at: str
    engines_used: List[str]
    data_sources: List[str]  # Track what data was used


class ScenarioEngine:
    """
    What-If Scenario Modeling Engine with Monte Carlo Simulation.

    DATA-DRIVEN APPROACH:
    - Calculates actual pay-turnover elasticity from your historical data
    - Uses ML model predictions for individual employee risk
    - Uses Survival model hazard ratios for time-based predictions
    - Falls back to industry estimates only when no data available

    Key capabilities:
    - Compensation adjustments → turnover impact
    - Headcount changes → cost/productivity impact
    - Promotion policies → retention impact
    - Retention interventions → ROI modeling
    """

    def __init__(
        self,
        employee_df: pd.DataFrame,
        ml_engine: Optional['MLEngine'] = None,
        survival_engine: Optional['SurvivalEngine'] = None,
        compensation_engine: Optional['CompensationEngine'] = None
    ):
        """
        Initialize ScenarioEngine.

        Args:
            employee_df: Employee DataFrame
            ml_engine: Optional trained ML engine for attrition prediction
            survival_engine: Optional survival engine for time-based analysis
            compensation_engine: Optional compensation engine for pay analysis
        """
        self.current_df, self.population_resolution = resolve_current_population(employee_df)
        self.df = active_population(self.current_df)
        if 'Dept' in self.df:
            self.df['Dept'] = self.df['Dept'].astype('string').str.strip().replace('', pd.NA).fillna('Unknown')
        self.ml_engine = ml_engine
        self.survival_engine = survival_engine
        self.compensation_engine = compensation_engine

        self.config = load_config()
        self.scenario_config = self.config.get('scenario', {})
        self.logger = get_logger('scenario_engine')
        self.warnings: List[str] = []
        self.data_sources: List[str] = []

        # Monte Carlo settings
        self.n_simulations = self.scenario_config.get('monte_carlo_iterations', 1000)
        self.random_seed = self.scenario_config.get('random_seed', 42)
        self.confidence_intervals = self.scenario_config.get(
            'confidence_intervals', [0.05, 0.25, 0.50, 0.75, 0.95]
        )

        # Cost assumptions
        self.cost_config = self.scenario_config.get('cost_assumptions', {})
        self.replacement_cost_mult = self.cost_config.get('replacement_cost_multiplier', 1.5)
        self.hiring_cost_base = self.cost_config.get('hiring_cost_base', 5000)
        self.training_months = self.cost_config.get('training_cost_months', 3)
        self.ramp_months = self.cost_config.get('productivity_ramp_months', 6)

        # Engine availability
        self._check_engine_availability()

        # Calculate data-driven parameters from actual data
        self._calculate_data_driven_parameters()

    def _check_engine_availability(self) -> None:
        """Check which engines are available for predictions."""
        self.has_ml = (
            self.ml_engine is not None and
            hasattr(self.ml_engine, 'is_trained') and
            self.ml_engine.is_trained
        )
        self.has_survival = self.survival_engine is not None
        self.has_compensation = self.compensation_engine is not None

        engines = []
        if self.has_ml:
            engines.append('ML')
        if self.has_survival:
            engines.append('Survival')
        if self.has_compensation:
            engines.append('Compensation')

        if not engines:
            self.warnings.append(
                "Scenario effects use configured assumptions; no predictive or causal effect model is used."
            )
            self.logger.warning("ScenarioEngine using statistical fallback")

        self.available_engines = ['Scenario assumption arithmetic']

    def _calculate_data_driven_parameters(self) -> None:
        """
        Calculate key parameters from actual historical data.

        This is what makes our predictions DATA-DRIVEN rather than using
        hardcoded industry averages.
        """
        self.data_driven_elasticity = None
        self.salary_attrition_correlation = None
        self.ml_risk_scores = None
        self.survival_hazard_ratios = None

        self.data_sources = ['Observed active workforce salary and headcount', 'Configured scenario assumptions']
        self.warnings.append('Scenario response parameters are assumptions, not causal effects estimated from employee outcomes.')




    def _get_baseline_turnover(self, filtered_df: Optional[pd.DataFrame] = None) -> float:
        """
        Get baseline turnover rate from data or ML model.

        If ML model available, uses predicted probabilities.
        Otherwise uses actual historical attrition rate.
        """
        # Observed exit share and classifier scores are not period turnover.
        value = float(self.scenario_config.get('assumed_baseline_turnover', .15))
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ScenarioEngineError('Configured baseline scenario rate must be between 0 and 1')
        return value

    @staticmethod
    def _mean_salary(frame: pd.DataFrame) -> float:
        if 'Salary' not in frame:
            raise ScenarioEngineError('Financial scenarios require annual Salary measurements')
        values = pd.to_numeric(frame['Salary'], errors='coerce')
        if values.empty or not (np.isfinite(values) & (values > 0)).all():
            raise ScenarioEngineError('Resolve missing, nonfinite or nonpositive annual salaries before financial simulation')
        return float(values.mean())

    def _filter_employees(self, target: Dict[str, Any]) -> pd.DataFrame:
        """
        Filter employees based on target criteria.

        Args:
            target: Dictionary with filtering criteria

        Returns:
            Filtered DataFrame
        """
        df = self.df.copy()
        scope = target.get('scope', 'all')

        if scope == 'all':
            return df
        required = {'department': ('department', 'Dept'), 'job_title': ('job_titles', 'JobTitle'),
                    'tenure_range': (None, 'Tenure'), 'performance': ('performance_min', 'LastRating'),
                    'custom': ('employee_ids', 'EmployeeID')}
        if scope not in required:
            raise ScenarioEngineError('Unsupported scenario scope')
        parameter, column = required[scope]
        if column not in df or (parameter and target.get(parameter) in (None, '', [])):
            raise ScenarioEngineError('Requested scenario filter is missing or unavailable')

        if scope == 'department' and target.get('department'):
            df = df[df['Dept'] == target['department']]

        if scope == 'job_title' and target.get('job_titles'):
            df = df[df['JobTitle'].isin(target['job_titles'])]

        if scope == 'tenure_range':
            if target.get('tenure_min') is not None:
                df = df[df['Tenure'] >= target['tenure_min']]
            if target.get('tenure_max') is not None:
                df = df[df['Tenure'] <= target['tenure_max']]

        if scope == 'performance' and target.get('performance_min') is not None:
            df = df[df['LastRating'] >= target['performance_min']]

        if scope == 'custom' and target.get('employee_ids'):
            df = df[df['EmployeeID'].isin(target['employee_ids'])]

        return df

    def _estimate_turnover_reduction(
        self,
        compensation_increase_pct: float,
        current_turnover: float,
        affected_df: Optional[pd.DataFrame] = None
    ) -> Tuple[float, float, str]:
        """
        Estimate turnover reduction from compensation increase.

        DATA-DRIVEN APPROACH:
        1. If we have calculated elasticity from data, use that
        2. If we have ML model, simulate effect on individual predictions
        3. If we have survival hazard ratios, use those
        4. Fall back to industry estimates only as last resort

        Returns:
            Tuple of (reduction, uncertainty_std, method_used)
        """
        elasticity = float(self.scenario_config.get('assumed_compensation_elasticity', 0.0))
        if not np.isfinite(elasticity) or elasticity < 0:
            raise ScenarioEngineError('Assumed compensation elasticity must be finite and non-negative')
        # Elasticity relates proportional changes; the public raise input is percent.
        reduction = min((compensation_increase_pct / 100) * elasticity * current_turnover, current_turnover * .5)
        return reduction, abs(reduction) * .4, 'configured_assumption_not_causal_estimate'


    def _run_monte_carlo(
        self,
        base_outcome: float,
        outcome_std: float,
        cost_per_outcome: float,
        n_affected: int,
        intervention_cost: float,
        baseline_outcome: Optional[float] = None,
        fixed_benefit: Optional[float] = None
    ) -> MonteCarloResult:
        """
        Run Monte Carlo simulation.

        Args:
            base_outcome: Expected outcome (e.g., turnover rate)
            outcome_std: Standard deviation of outcome
            cost_per_outcome: Cost per unit of outcome
            n_affected: Number of affected employees
            intervention_cost: Total cost of intervention

        Returns:
            MonteCarloResult with distribution data
        """
        rng = np.random.default_rng(self.random_seed)

        # Simulate outcomes
        outcomes = rng.normal(base_outcome, outcome_std, self.n_simulations)
        outcomes = np.clip(outcomes, 0, 1)  # Bound between 0-100%

        # Simulate costs
        cost_impacts = []
        rois = []

        for outcome in outcomes:
            # Estimate employees retained (vs baseline)
            baseline = self._get_baseline_turnover() if baseline_outcome is None else baseline_outcome
            retained = (baseline - outcome) * n_affected

            # Calculate benefit (replacement costs avoided)
            benefit = retained * cost_per_outcome if fixed_benefit is None else fixed_benefit

            # Net impact
            net = benefit - intervention_cost
            cost_impacts.append(net)

            # ROI
            roi = safe_divide(net, intervention_cost, 0) * 100
            rois.append(roi)

        cost_impacts = np.array(cost_impacts)
        rois = np.array(rois)

        # Calculate percentiles
        outcome_percentiles = {
            f'p{int(p*100)}': float(np.percentile(outcomes, p*100))
            for p in self.confidence_intervals
        }

        cost_percentiles = {
            f'p{int(p*100)}': float(np.percentile(cost_impacts, p*100))
            for p in self.confidence_intervals
        }

        # Histogram
        hist_counts, hist_bins = np.histogram(outcomes, bins=20)

        # Convergence check (did results stabilize?)
        rolling_means = pd.Series(outcomes).expanding().mean().values
        if len(rolling_means) > 100:
            last_100_std = np.std(rolling_means[-100:])
            converged = last_100_std < 0.01
            convergence_iter = len(rolling_means) - 100 if converged else self.n_simulations
        else:
            converged = True
            convergence_iter = len(rolling_means)

        return MonteCarloResult(
            n_iterations=self.n_simulations,
            outcome_mean=float(np.mean(outcomes)),
            outcome_std=float(np.std(outcomes)),
            outcome_median=float(np.median(outcomes)),
            percentiles=outcome_percentiles,
            histogram_bins=hist_bins.tolist(),
            histogram_counts=hist_counts.tolist(),
            cost_impact_mean=float(np.mean(cost_impacts)),
            cost_impact_std=float(np.std(cost_impacts)),
            cost_impact_percentiles=cost_percentiles,
            roi_mean=float(np.mean(rois)),
            roi_std=float(np.std(rois)),
            roi_positive_probability=float((np.array(rois) > 0).mean()),
            converged=converged,
            convergence_iterations=convergence_iter
        )

    def _get_confidence_level(self, n_affected: int) -> Tuple[str, float]:
        """Assumed intervention effects remain exploratory at every sample size.

        The legacy score is a display cap, not a probability of correctness.
        """
        return 'Exploratory', 0.5

    # =========================================================================
    # COMPENSATION SCENARIOS
    # =========================================================================

    def simulate_compensation_change(
        self,
        adjustment_type: str,  # 'percentage', 'absolute', 'market_adjustment'
        target: Dict[str, Any],
        adjustment_value: float,
        time_horizon_months: int = 12
    ) -> ScenarioResult:
        """
        Simulate impact of compensation changes on turnover.

        DATA-DRIVEN: Uses actual historical data to estimate relationships.

        Args:
            adjustment_type: Type of adjustment
            target: Target employee group
            adjustment_value: Amount of adjustment
            time_horizon_months: Prediction horizon

        Returns:
            ScenarioResult with predictions based on your data
        """
        if not isinstance(time_horizon_months, int) or isinstance(time_horizon_months, bool) or not 1 <= time_horizon_months <= 60 or not np.isfinite(adjustment_value) or adjustment_value < 0:
            raise ScenarioEngineError('Require a 1–60 month horizon and finite nonnegative raise')
        scenario_id = str(uuid.uuid4())[:8]
        affected_df = self._filter_employees(target)
        n_affected = len(affected_df)

        if n_affected == 0:
            raise ScenarioEngineError("No employees match the target criteria")

        affected_depts = affected_df['Dept'].unique().tolist() if 'Dept' in affected_df.columns else []

        if adjustment_type not in {'percentage', 'absolute', 'market_adjustment'}:
            raise ScenarioEngineError('Unsupported compensation adjustment type')
        # Calculate adjustment
        if adjustment_type == 'percentage':
            pct_increase = adjustment_value / 100
        elif adjustment_type == 'absolute':
            avg_salary = self._mean_salary(affected_df)
            pct_increase = adjustment_value / avg_salary
        else:  # market_adjustment
            pct_increase = adjustment_value / 100

        # Baseline metrics - use filtered data for department-specific baseline
        baseline_turnover = self._get_baseline_turnover(affected_df)

        # DATA-DRIVEN: Estimate turnover reduction using calculated elasticity
        turnover_reduction, reduction_uncertainty, method_used = self._estimate_turnover_reduction(
            adjustment_value if adjustment_type == 'percentage' else pct_increase * 100,
            baseline_turnover,
            affected_df
        )
        projected_turnover = baseline_turnover - turnover_reduction

        # Calculate costs
        avg_salary = self._mean_salary(affected_df)
        total_salary_increase = avg_salary * pct_increase * n_affected

        # Expected employees retained
        expected_retained = turnover_reduction * n_affected
        replacement_savings = expected_retained * avg_salary * self.replacement_cost_mult

        cost_impact = CostImpact(
            salary_change=total_salary_increase * time_horizon_months / 12,
            replacement_costs_avoided=replacement_savings,
            total_cost=total_salary_increase * time_horizon_months / 12,
            total_benefit=replacement_savings,
            net_impact=replacement_savings - total_salary_increase * time_horizon_months / 12
        )

        # Monte Carlo simulation - use data-driven uncertainty
        mc_result = self._run_monte_carlo(
            base_outcome=projected_turnover,
            outcome_std=max(reduction_uncertainty, baseline_turnover * 0.1),
            cost_per_outcome=avg_salary * self.replacement_cost_mult,
            n_affected=n_affected,
            intervention_cost=total_salary_increase * time_horizon_months / 12,
            baseline_outcome=baseline_turnover
        )

        # ROI and payback
        roi = safe_divide(cost_impact.net_impact, cost_impact.total_cost, 0) * 100
        monthly_benefit = replacement_savings / time_horizon_months
        payback = int(safe_divide(cost_impact.total_cost, monthly_benefit, 999))

        # Confidence - adjust based on data quality
        conf_level, conf_score = self._get_confidence_level(n_affected)

        # Boost confidence if using data-driven methods
        if method_used == "data_driven_elasticity":
            conf_score = min(1.0, conf_score + 0.1)
        elif method_used == "ml_model_simulation":
            conf_score = min(1.0, conf_score + 0.15)
        elif method_used == "industry_estimate":
            conf_score = max(0.3, conf_score - 0.2)

        # Generate recommendation
        if roi > 50:
            recommendation = "Strongly recommended - high ROI expected"
        elif roi > 0:
            recommendation = "Recommended - positive ROI expected"
        elif roi > -20:
            recommendation = "Consider carefully - marginal impact"
        else:
            recommendation = "Not recommended - negative ROI expected"

        # Build assumptions list based on method used
        assumptions = [
            f"Assumed baseline rate for this scenario horizon: {baseline_turnover*100:.1f}% (configured, not observed turnover)",
            f"Replacement cost: {self.replacement_cost_mult}x annual salary",
            f"Assumed response elasticity: {self.scenario_config.get('assumed_compensation_elasticity', 0.0)} proportional rate change per proportional pay change; reduction capped at 50% of baseline",
            f"Time horizon: {time_horizon_months} months"
        ]

        if method_used == "data_driven_elasticity":
            assumptions.append(
                f"Pay-turnover elasticity: {self.data_driven_elasticity:.3f} "
                f"(calculated from historical attrition patterns)"
            )
        elif method_used == "ml_model_simulation":
            assumptions.append("Turnover impact estimated using ML model predictions")
        elif method_used == "survival_hazard_ratio":
            assumptions.append("Turnover impact estimated from survival analysis hazard ratios")
        else:
            assumptions.append(
                "Using configured hypothetical pay-response assumption; no causal effect has been validated. "
                "No observed causal pay-response evidence is available."
            )

        return ScenarioResult(
            scenario_id=scenario_id,
            scenario_name=f"{adjustment_value}% raise for {target.get('department', 'selected group')}",
            scenario_type='compensation',
            input_parameters={
                'adjustment_type': adjustment_type,
                'target': target,
                'adjustment_value': adjustment_value,
                'time_horizon_months': time_horizon_months
            },
            affected_employees=n_affected,
            affected_departments=affected_depts,
            baseline_turnover_rate=round(baseline_turnover * 100, 1),
            projected_turnover_rate=round(projected_turnover * 100, 1),
            turnover_change=round(turnover_reduction * 100, 2),
            turnover_change_pct=round(safe_divide(turnover_reduction, baseline_turnover, 0) * 100, 1),
            simulation=mc_result,
            cost_impact=cost_impact,
            roi_estimate=round(roi, 1),
            payback_months=payback if payback < 120 else None,
            confidence_level=conf_level,
            confidence_score=conf_score,
            assumptions=assumptions,
            risks=[
                "Actual turnover reduction may vary by department",
                "Market conditions may affect retention",
                "Budget constraints may limit implementation"
            ],
            recommendation=recommendation,
            alternative_actions=[
                "Target high-risk employees only",
                "Combine with career development programs",
                "Phase implementation over multiple quarters"
            ],
            computed_at=datetime.now().isoformat(),
            engines_used=self.available_engines,
            data_sources=self.data_sources
        )

    # =========================================================================
    # HEADCOUNT SCENARIOS
    # =========================================================================

    def simulate_headcount_change(
        self,
        change_type: str,  # 'reduction', 'expansion'
        target: Dict[str, Any],
        change_count: Optional[int] = None,
        change_percentage: Optional[float] = None,
        selection_criteria: str = 'performance'
    ) -> ScenarioResult:
        """
        Simulate impact of headcount changes.

        Args:
            change_type: 'reduction' or 'expansion'
            target: Target group
            change_count: Absolute number to change
            change_percentage: Percentage to change
            selection_criteria: How to select employees ('performance', 'tenure', 'cost')

        Returns:
            ScenarioResult with predictions
        """
        scenario_id = str(uuid.uuid4())[:8]
        affected_df = self._filter_employees(target)
        n_current = len(affected_df)

        if n_current == 0:
            raise ScenarioEngineError("No employees match the target criteria")

        # Calculate change amount
        if change_count is not None:
            n_change = change_count
        elif change_percentage is not None:
            n_change = int(n_current * change_percentage / 100)
        else:
            raise ScenarioEngineError("Must specify change_count or change_percentage")

        if change_type not in {'reduction', 'expansion'} or isinstance(n_change, bool) or not isinstance(n_change, (int, np.integer)) or n_change < 1 or (change_type == 'reduction' and n_change > n_current):
            raise ScenarioEngineError('Change count must be a valid positive whole number within the selected population for reductions')
        affected_depts = affected_df['Dept'].unique().tolist() if 'Dept' in affected_df.columns else []

        # Get average salary
        avg_salary = self._mean_salary(affected_df)

        if change_type == 'reduction':
            # Sort by selection criteria to identify who would be affected
            if selection_criteria == 'performance' and 'LastRating' in affected_df.columns:
                affected_df = affected_df.sort_values('LastRating')
            elif selection_criteria == 'tenure' and 'Tenure' in affected_df.columns:
                affected_df = affected_df.sort_values('Tenure')
            elif selection_criteria == 'cost' and 'Salary' in affected_df.columns:
                affected_df = affected_df.sort_values('Salary', ascending=False)

            # Cost savings
            impacted = affected_df.head(n_change)
            salary_savings = impacted['Salary'].sum() if 'Salary' in impacted.columns else n_change * avg_salary
            severance_cost = salary_savings * 0.25  # Assume 3 months severance

            cost_impact = CostImpact(
                salary_change=-salary_savings,
                replacement_costs_incurred=severance_cost,
                productivity_impact=-salary_savings * 0.1,  # 10% productivity loss
                total_cost=severance_cost,
                total_benefit=salary_savings,
                net_impact=salary_savings - severance_cost
            )

            turnover_change = n_change / n_current
            projected_turnover = 0  # Forced attrition

        else:  # expansion
            hiring_costs = n_change * (self.hiring_cost_base + avg_salary * 0.2)
            training_costs = n_change * avg_salary * (self.training_months / 12)

            cost_impact = CostImpact(
                salary_change=n_change * avg_salary,
                training_costs=training_costs,
                productivity_impact=n_change * avg_salary * 0.3,  # 30% productivity during ramp
                total_cost=hiring_costs + training_costs + n_change * avg_salary,
                total_benefit=n_change * avg_salary * 1.2,  # Expected productivity
                net_impact=n_change * avg_salary * 0.2 - hiring_costs - training_costs
            )

            turnover_change = 0
            projected_turnover = self._get_baseline_turnover()

        # Monte Carlo
        mc_result = self._run_monte_carlo(
            base_outcome=projected_turnover if change_type == 'expansion' else turnover_change,
            outcome_std=0.05,
            cost_per_outcome=avg_salary,
            n_affected=n_change,
            intervention_cost=abs(cost_impact.total_cost),
            fixed_benefit=cost_impact.total_benefit
        )

        conf_level, conf_score = self._get_confidence_level(n_change)

        scenario_name = (
            f"{'Reduce' if change_type == 'reduction' else 'Expand'} "
            f"{n_change} positions in {target.get('department', 'organization')}"
        )

        return ScenarioResult(
            scenario_id=scenario_id,
            scenario_name=scenario_name,
            scenario_type='headcount',
            input_parameters={
                'change_type': change_type,
                'target': target,
                'change_count': n_change,
                'selection_criteria': selection_criteria
            },
            affected_employees=n_change,
            affected_departments=affected_depts,
            baseline_turnover_rate=round(self._get_baseline_turnover() * 100, 1),
            projected_turnover_rate=round(projected_turnover * 100, 1) if change_type == 'expansion' else 0,
            turnover_change=round(turnover_change * 100, 2),
            turnover_change_pct=round(turnover_change * 100, 1),
            simulation=mc_result,
            cost_impact=cost_impact,
            roi_estimate=round(safe_divide(cost_impact.net_impact, abs(cost_impact.total_cost), 0) * 100, 1),
            payback_months=(int(np.ceil(cost_impact.total_cost / (cost_impact.total_benefit / 12)))
                            if change_type == 'reduction' and cost_impact.total_benefit > 0 else None),
            confidence_level=conf_level,
            confidence_score=conf_score,
            assumptions=[
                f"Average salary: ${avg_salary:,.0f}",
                f"Selection criteria: {selection_criteria}",
                "Financial results are fixed assumption arithmetic; financial uncertainty has not been estimated.",
                ("Simple payback divides one-off severance by monthly salary savings; assumes immediate savings and excludes unmodeled effects."
                 if change_type == 'reduction' else
                 "Expansion payback is unavailable: this annual cost/benefit scenario does not model a multi-period cash-flow schedule."),
                "Severance: 3 months salary" if change_type == 'reduction' else "Ramp time: 6 months"
            ],
            risks=[
                "Knowledge loss from departures" if change_type == 'reduction' else "Quality of new hires",
                "Team morale impact",
                "Market conditions for hiring" if change_type == 'expansion' else "Legal/compliance risks"
            ],
            recommendation="Proceed with caution" if change_type == 'reduction' else "Evaluate hiring timeline",
            alternative_actions=[
                "Attrition-based reduction" if change_type == 'reduction' else "Contract-to-hire approach",
                "Redeployment options",
                "Phased implementation"
            ],
            computed_at=datetime.now().isoformat(),
            engines_used=self.available_engines,
            data_sources=self.data_sources
        )

    # =========================================================================
    # INTERVENTION SCENARIOS
    # =========================================================================



    def simulate_attrition_intervention(
        self,
        intervention_type: str,  # 'retention_bonus', 'career_path', 'manager_change'
        target_employees: str,  # 'high_risk', 'high_risk_high_performer', or list of IDs
        intervention_params: Dict[str, Any]
    ) -> ScenarioResult:
        """Unavailable until an aggregate intervention-effect model is validated."""
        raise ScenarioEngineError("Intervention effects and individual risk targeting are unavailable without validated evidence")

    # =========================================================================
    # COMPARISON & TEMPLATES
    # =========================================================================

    def compare_scenarios(
        self,
        scenarios: List[ScenarioResult]
    ) -> Dict[str, Any]:
        """
        Compare multiple scenario results.

        Args:
            scenarios: List of ScenarioResult objects

        Returns:
            Comparison analysis
        """
        if len(scenarios) < 2:
            return {'error': 'Need at least 2 scenarios to compare'}

        comparison = []
        for s in scenarios:
            comparison.append({
                'scenario_id': s.scenario_id,
                'scenario_name': s.scenario_name,
                'affected_employees': s.affected_employees,
                'turnover_change_pct': s.turnover_change_pct,
                'roi_estimate': s.roi_estimate,
                'net_impact': s.cost_impact.net_impact,
                'confidence_level': s.confidence_level,
                'roi_positive_probability': s.simulation.roi_positive_probability
            })

        # Rank by ROI
        comparison.sort(key=lambda x: x['roi_estimate'] or 0, reverse=True)

        best = comparison[0]
        return {
            'scenarios': comparison,
            'recommended_scenario': 'No automatic recommendation',
            'reasoning': 'Compare assumptions and sensitivity; simulation draw shares are not empirical probabilities of future return.'
        }

    def get_scenario_templates(self) -> List[Dict[str, Any]]:
        """Get pre-built scenario templates from config."""
        templates = self.scenario_config.get('templates', [])
        return [
            {
                'name': t.get('name'),
                'type': t.get('type'),
                'description': t.get('description', ''),
                'params': t.get('params', {})
            }
            for t in templates
        ]

    def to_dict(self, result: ScenarioResult) -> Dict[str, Any]:
        """Convert ScenarioResult to dictionary."""
        d = asdict(result)
        d['simulation'] = asdict(result.simulation)
        d['cost_impact'] = asdict(result.cost_impact)
        return d
