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
from scipy import stats

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
        self.df = employee_df.copy()
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
                "No ML/Survival engines available. Using statistical models from your data."
            )
            self.logger.warning("ScenarioEngine using statistical fallback")

        self.available_engines = engines

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

        # 1. Calculate pay-turnover elasticity from actual data
        if 'Attrition' in self.df.columns and 'Salary' in self.df.columns:
            self._calculate_elasticity_from_data()
            self.data_sources.append('Historical attrition data')

        # 2. Get ML model risk predictions if available
        if self.has_ml:
            self._get_ml_predictions()
            self.data_sources.append('ML model predictions')

        # 3. Get survival hazard ratios if available
        if self.has_survival:
            self._get_survival_hazards()
            self.data_sources.append('Survival analysis hazard ratios')

        # 4. Get compensation analysis if available
        if self.has_compensation:
            self.data_sources.append('Compensation analysis')

        if not self.data_sources:
            self.data_sources.append('Industry estimates (no historical data)')
            self.warnings.append(
                "Using industry estimates. Upload data with Attrition column for data-driven predictions."
            )

    def _calculate_elasticity_from_data(self) -> None:
        """
        Calculate actual pay-turnover elasticity from historical data.

        Elasticity = % change in turnover / % change in pay

        This replaces the hardcoded 0.02 with actual data-driven value.
        """
        try:
            df = self.df.copy()

            # Ensure numeric types
            df['Salary'] = pd.to_numeric(df['Salary'], errors='coerce')
            df['Attrition'] = pd.to_numeric(df['Attrition'], errors='coerce')

            # Filter valid rows
            valid = df[['Salary', 'Attrition']].dropna()
            if len(valid) < 30:
                self.logger.warning("Insufficient data for elasticity calculation")
                return

            # Method 1: Direct correlation
            correlation = valid['Salary'].corr(valid['Attrition'])
            self.salary_attrition_correlation = correlation

            # Method 2: Binned analysis (more robust)
            # Group by salary quartiles and calculate attrition rate in each
            valid['salary_quartile'] = pd.qcut(
                valid['Salary'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4']
            )

            quartile_rates = valid.groupby('salary_quartile')['Attrition'].mean()

            # Calculate elasticity: how much does attrition change per salary change
            if len(quartile_rates) >= 2:
                # Compare top vs bottom quartile
                low_salary_attrition = quartile_rates.iloc[0]  # Q1 (lowest salary)
                high_salary_attrition = quartile_rates.iloc[-1]  # Q4 (highest salary)

                # Get median salaries for each quartile
                salary_quartiles = valid.groupby('salary_quartile')['Salary'].median()
                low_salary = salary_quartiles.iloc[0]
                high_salary = salary_quartiles.iloc[-1]

                if high_salary > low_salary and low_salary_attrition > 0:
                    # Calculate elasticity
                    pct_salary_diff = (high_salary - low_salary) / low_salary
                    pct_attrition_diff = (low_salary_attrition - high_salary_attrition) / low_salary_attrition

                    if pct_salary_diff > 0:
                        self.data_driven_elasticity = pct_attrition_diff / pct_salary_diff

                        self.logger.info(
                            f"Data-driven elasticity: {self.data_driven_elasticity:.4f} "
                            f"(Q1 attrition: {low_salary_attrition:.1%}, "
                            f"Q4 attrition: {high_salary_attrition:.1%})"
                        )

            # Method 3: Logistic regression coefficient for salary
            if len(valid) >= 50:
                try:
                    from sklearn.linear_model import LogisticRegression
                    from sklearn.preprocessing import StandardScaler

                    X = valid[['Salary']].values
                    y = valid['Attrition'].values

                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)

                    model = LogisticRegression(random_state=42)
                    model.fit(X_scaled, y)

                    # Coefficient tells us relationship direction and strength
                    self.salary_coefficient = model.coef_[0][0]
                    self.logger.info(f"Logistic regression salary coefficient: {self.salary_coefficient:.4f}")

                except Exception as e:
                    self.logger.warning(f"Logistic regression failed: {e}")

        except Exception as e:
            self.logger.error(f"Elasticity calculation error: {e}")

    def _get_ml_predictions(self) -> None:
        """Get ML model predictions for each employee."""
        try:
            if hasattr(self.ml_engine, 'predict_proba'):
                # Get predictions for all employees
                predictions = self.ml_engine.predict_proba(self.df)
                if predictions is not None:
                    self.ml_risk_scores = predictions
                    self.df['_ml_risk_score'] = predictions
                    self.logger.info(f"ML predictions loaded for {len(predictions)} employees")

            elif hasattr(self.ml_engine, 'get_predictions'):
                result = self.ml_engine.get_predictions()
                if result and 'predictions' in result:
                    self.ml_risk_scores = result['predictions']

        except Exception as e:
            self.logger.error(f"ML predictions error: {e}")

    def _get_survival_hazards(self) -> None:
        """Get survival model hazard ratios."""
        try:
            if hasattr(self.survival_engine, 'get_cox_model'):
                cox_result = self.survival_engine.get_cox_model()
                if cox_result and 'hazard_ratios' in cox_result:
                    self.survival_hazard_ratios = cox_result['hazard_ratios']
                    self.logger.info("Survival hazard ratios loaded")

                    # Look for salary-related hazard ratio
                    for hr in self.survival_hazard_ratios:
                        if 'salary' in hr.get('factor', '').lower():
                            self.salary_hazard_ratio = hr.get('hazard_ratio', 1.0)

        except Exception as e:
            self.logger.error(f"Survival hazards error: {e}")

    def _get_baseline_turnover(self, filtered_df: Optional[pd.DataFrame] = None) -> float:
        """
        Get baseline turnover rate from data or ML model.

        If ML model available, uses predicted probabilities.
        Otherwise uses actual historical attrition rate.
        """
        df = filtered_df if filtered_df is not None else self.df

        # If we have ML predictions, use those
        if '_ml_risk_score' in df.columns:
            return float(df['_ml_risk_score'].mean())

        # Otherwise use actual attrition data
        if 'Attrition' in df.columns:
            return float(df['Attrition'].mean())

        return 0.15  # Default assumption only if no data

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
        method_used = "industry_estimate"
        uncertainty = 0.3  # Default uncertainty

        # Method 1: Use data-driven elasticity (BEST)
        if self.data_driven_elasticity is not None:
            elasticity = self.data_driven_elasticity
            method_used = "data_driven_elasticity"
            # Lower uncertainty because it's from actual data
            uncertainty = 0.15

            self.logger.info(
                f"Using data-driven elasticity: {elasticity:.4f} "
                f"(calculated from historical turnover patterns)"
            )

        # Method 2: Use ML model to simulate individual effects
        elif self.has_ml and affected_df is not None and '_ml_risk_score' in affected_df.columns:
            # Simulate how salary increase would change individual risk scores
            # This uses the model's learned relationship
            elasticity = self._estimate_elasticity_from_ml(affected_df, compensation_increase_pct)
            method_used = "ml_model_simulation"
            uncertainty = 0.2

        # Method 3: Use survival hazard ratio for salary
        elif hasattr(self, 'salary_hazard_ratio'):
            # Convert hazard ratio to elasticity
            # HR < 1 means higher salary = lower hazard (lower turnover)
            hr = self.salary_hazard_ratio
            if hr < 1 and hr > 0:
                # Rough conversion: elasticity ≈ (1 - HR) / 10
                elasticity = (1 - hr) / 10
            else:
                elasticity = 0.02
            method_used = "survival_hazard_ratio"
            uncertainty = 0.25

        # Method 4: Use correlation as rough estimate
        elif self.salary_attrition_correlation is not None:
            # Negative correlation means higher salary = lower attrition
            corr = self.salary_attrition_correlation
            # Convert to rough elasticity
            elasticity = abs(corr) * 0.05 if corr < 0 else 0.01
            method_used = "salary_attrition_correlation"
            uncertainty = 0.25

        # Method 5: Fall back to industry estimate (WORST)
        else:
            elasticity = 0.02  # Conservative industry estimate
            method_used = "industry_estimate"
            uncertainty = 0.4  # High uncertainty

            self.warnings.append(
                "Using industry estimate (2% reduction per 1% raise). "
                "Add Attrition column to your data for data-driven predictions."
            )

        # Calculate reduction
        reduction = compensation_increase_pct * elasticity * current_turnover

        # Cap at 50% maximum reduction (can't eliminate all turnover)
        reduction = min(reduction, current_turnover * 0.5)

        # Calculate uncertainty in terms of standard deviation
        uncertainty_std = reduction * uncertainty

        self.logger.info(
            f"Turnover reduction estimate: {reduction:.4f} (±{uncertainty_std:.4f}) "
            f"using method: {method_used}"
        )

        return reduction, uncertainty_std, method_used

    def _estimate_elasticity_from_ml(
        self,
        affected_df: pd.DataFrame,
        pct_increase: float
    ) -> float:
        """
        Estimate elasticity by simulating ML model predictions with salary change.

        This is a sophisticated approach that uses the trained model to
        estimate how individual risk scores would change.
        """
        try:
            if not self.has_ml:
                return 0.02

            # Get current average risk
            current_risk = affected_df['_ml_risk_score'].mean()

            # Create modified DataFrame with increased salary
            modified_df = affected_df.copy()
            if 'Salary' in modified_df.columns:
                modified_df['Salary'] = modified_df['Salary'] * (1 + pct_increase / 100)

                # Get new predictions
                new_predictions = self.ml_engine.predict_proba(modified_df)
                if new_predictions is not None:
                    new_risk = new_predictions.mean()

                    # Calculate implied elasticity
                    if current_risk > 0 and pct_increase > 0:
                        risk_change_pct = (current_risk - new_risk) / current_risk
                        implied_elasticity = risk_change_pct / pct_increase

                        self.logger.info(
                            f"ML-simulated elasticity: {implied_elasticity:.4f} "
                            f"(risk changed from {current_risk:.3f} to {new_risk:.3f})"
                        )
                        return max(0, implied_elasticity)

        except Exception as e:
            self.logger.warning(f"ML elasticity simulation failed: {e}")

        return 0.02  # Fallback

    def _run_monte_carlo(
        self,
        base_outcome: float,
        outcome_std: float,
        cost_per_outcome: float,
        n_affected: int,
        intervention_cost: float
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
        np.random.seed(self.random_seed)

        # Simulate outcomes
        outcomes = np.random.normal(base_outcome, outcome_std, self.n_simulations)
        outcomes = np.clip(outcomes, 0, 1)  # Bound between 0-100%

        # Simulate costs
        cost_impacts = []
        rois = []

        for outcome in outcomes:
            # Estimate employees retained (vs baseline)
            baseline = self._get_baseline_turnover()
            retained = (baseline - outcome) * n_affected

            # Calculate benefit (replacement costs avoided)
            avg_salary = self.df['Salary'].mean() if 'Salary' in self.df.columns else 75000
            benefit = retained * avg_salary * self.replacement_cost_mult

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
        """Determine confidence level based on sample size."""
        conf_config = self.scenario_config.get('confidence', {})
        high_threshold = conf_config.get('min_sample_for_high', 100)
        medium_threshold = conf_config.get('min_sample_for_medium', 30)

        if n_affected >= high_threshold:
            return 'High', 0.9
        elif n_affected >= medium_threshold:
            return 'Medium', 0.7
        else:
            return 'Low', 0.5

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
        scenario_id = str(uuid.uuid4())[:8]
        affected_df = self._filter_employees(target)
        n_affected = len(affected_df)

        if n_affected == 0:
            raise ScenarioEngineError("No employees match the target criteria")

        affected_depts = affected_df['Dept'].unique().tolist() if 'Dept' in affected_df.columns else []

        # Calculate adjustment
        if adjustment_type == 'percentage':
            pct_increase = adjustment_value / 100
        elif adjustment_type == 'absolute':
            avg_salary = affected_df['Salary'].mean() if 'Salary' in affected_df.columns else 75000
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
        avg_salary = affected_df['Salary'].mean() if 'Salary' in affected_df.columns else 75000
        total_salary_increase = avg_salary * pct_increase * n_affected

        # Expected employees retained
        expected_retained = turnover_reduction * n_affected
        replacement_savings = expected_retained * avg_salary * self.replacement_cost_mult

        cost_impact = CostImpact(
            salary_change=total_salary_increase * 12,  # Annualized
            replacement_costs_avoided=replacement_savings,
            total_cost=total_salary_increase * 12,
            total_benefit=replacement_savings,
            net_impact=replacement_savings - total_salary_increase * 12
        )

        # Monte Carlo simulation - use data-driven uncertainty
        mc_result = self._run_monte_carlo(
            base_outcome=projected_turnover,
            outcome_std=max(reduction_uncertainty, baseline_turnover * 0.1),
            cost_per_outcome=avg_salary * self.replacement_cost_mult,
            n_affected=n_affected,
            intervention_cost=total_salary_increase * 12
        )

        # ROI and payback
        roi = safe_divide(cost_impact.net_impact, cost_impact.total_cost, 0) * 100
        monthly_benefit = replacement_savings / 12
        payback = int(safe_divide(total_salary_increase, monthly_benefit, 999))

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
            f"Baseline turnover rate: {baseline_turnover*100:.1f}% (from your data)",
            f"Replacement cost: {self.replacement_cost_mult}x annual salary",
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
                "Using industry estimate for pay-turnover relationship "
                "(add Attrition column for data-driven predictions)"
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

        affected_depts = affected_df['Dept'].unique().tolist() if 'Dept' in affected_df.columns else []

        # Get average salary
        avg_salary = affected_df['Salary'].mean() if 'Salary' in affected_df.columns else 75000

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
            intervention_cost=abs(cost_impact.total_cost)
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
            payback_months=int(safe_divide(abs(cost_impact.total_cost), abs(cost_impact.net_impact) / 12, 999)),
            confidence_level=conf_level,
            confidence_score=conf_score,
            assumptions=[
                f"Average salary: ${avg_salary:,.0f}",
                f"Selection criteria: {selection_criteria}",
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

    def _identify_high_risk_employees(
        self,
        filter_type: str = 'high_risk'
    ) -> pd.DataFrame:
        """
        DATA-DRIVEN: Identify high-risk employees using ML predictions or data analysis.

        This replaces hardcoded risk identification with actual predictions.
        """
        df = self.df.copy()

        # Method 1: Use ML predictions if available (BEST)
        if '_ml_risk_score' in df.columns:
            if filter_type == 'high_risk':
                # Top 20% by predicted risk
                threshold = df['_ml_risk_score'].quantile(0.8)
                return df[df['_ml_risk_score'] >= threshold]
            elif filter_type == 'high_risk_high_performer':
                # High risk AND high performer
                risk_threshold = df['_ml_risk_score'].quantile(0.7)
                perf_threshold = df['LastRating'].quantile(0.7) if 'LastRating' in df.columns else 4
                return df[
                    (df['_ml_risk_score'] >= risk_threshold) &
                    (df.get('LastRating', 3) >= perf_threshold)
                ]

        # Method 2: Use survival analysis if available
        if self.has_survival and hasattr(self, 'survival_hazard_ratios'):
            # Would use survival predictions here
            pass

        # Method 3: Use historical attrition patterns
        if 'Attrition' in df.columns:
            # Identify characteristics of past leavers
            leavers = df[df['Attrition'] == 1]
            if len(leavers) > 10:
                # Find employees similar to leavers
                # (simplified - in production would use clustering/similarity)
                if 'Tenure' in df.columns:
                    avg_leaver_tenure = leavers['Tenure'].mean()
                    # Employees with similar tenure are higher risk
                    tenure_risk = abs(df['Tenure'] - avg_leaver_tenure) < leavers['Tenure'].std()
                    df['_tenure_risk'] = tenure_risk

                if 'Salary' in df.columns:
                    avg_leaver_salary = leavers['Salary'].mean()
                    # Employees with similar salary profile
                    df['_salary_risk'] = df['Salary'] < avg_leaver_salary

                # Combine risk factors
                risk_cols = [c for c in df.columns if c.endswith('_risk')]
                if risk_cols:
                    df['_composite_risk'] = df[risk_cols].sum(axis=1)
                    threshold = df['_composite_risk'].quantile(0.7)
                    return df[df['_composite_risk'] >= threshold]

        # Method 4: Fallback to heuristics (WORST)
        self.warnings.append(
            "Using heuristic risk identification. "
            "Add Attrition column or ML model for data-driven targeting."
        )

        if filter_type == 'high_risk':
            # Simple heuristic: low tenure + low rating
            conditions = pd.Series([True] * len(df), index=df.index)
            if 'Tenure' in df.columns:
                conditions &= df['Tenure'] < df['Tenure'].median()
            if 'LastRating' in df.columns:
                conditions &= df['LastRating'] < df['LastRating'].median()
            return df[conditions].head(int(len(df) * 0.2))

        elif filter_type == 'high_risk_high_performer':
            if 'LastRating' in df.columns:
                return df[df['LastRating'] >= 4].head(int(len(df) * 0.1))

        return df.head(int(len(df) * 0.2))

    def _calculate_intervention_effectiveness(
        self,
        intervention_type: str,
        target_df: pd.DataFrame
    ) -> Tuple[float, str]:
        """
        DATA-DRIVEN: Calculate intervention effectiveness from historical data.

        If we have data on past interventions, use that.
        Otherwise use research-based estimates.
        """
        method_used = "research_estimate"

        # Check if we have historical intervention data
        # (This would require intervention history columns in the data)
        if 'PastRetentionBonus' in self.df.columns and 'Attrition' in self.df.columns:
            # Calculate actual effectiveness from history
            bonus_received = self.df[self.df['PastRetentionBonus'] == 1]
            no_bonus = self.df[self.df['PastRetentionBonus'] == 0]

            if len(bonus_received) > 20 and len(no_bonus) > 20:
                bonus_attrition = bonus_received['Attrition'].mean()
                no_bonus_attrition = no_bonus['Attrition'].mean()

                if no_bonus_attrition > 0:
                    effectiveness = (no_bonus_attrition - bonus_attrition) / no_bonus_attrition
                    method_used = "historical_data"
                    return max(0, min(0.7, effectiveness)), method_used

        # Use research-based estimates with variation
        base_effectiveness = {
            'retention_bonus': 0.35,  # Research shows 25-45% effectiveness
            'career_path': 0.30,      # Development programs ~25-35%
            'manager_change': 0.25,   # Manager changes ~20-30%
        }

        effectiveness = base_effectiveness.get(intervention_type, 0.25)

        # Adjust based on target group characteristics
        if '_ml_risk_score' in target_df.columns:
            avg_risk = target_df['_ml_risk_score'].mean()
            # Higher risk employees may respond more to interventions
            if avg_risk > 0.6:
                effectiveness *= 1.2
            elif avg_risk < 0.3:
                effectiveness *= 0.8

        return min(0.7, effectiveness), method_used

    def simulate_attrition_intervention(
        self,
        intervention_type: str,  # 'retention_bonus', 'career_path', 'manager_change'
        target_employees: str,  # 'high_risk', 'high_risk_high_performer', or list of IDs
        intervention_params: Dict[str, Any]
    ) -> ScenarioResult:
        """
        Simulate retention intervention effectiveness.

        DATA-DRIVEN: Uses ML predictions and historical data when available.

        Args:
            intervention_type: Type of intervention
            target_employees: Who to target
            intervention_params: Parameters for the intervention

        Returns:
            ScenarioResult with predictions
        """
        scenario_id = str(uuid.uuid4())[:8]

        # DATA-DRIVEN: Identify target employees using ML or data analysis
        if target_employees in ['high_risk', 'high_risk_high_performer']:
            target_df = self._identify_high_risk_employees(target_employees)
        elif isinstance(target_employees, list):
            target_df = self.df[self.df['EmployeeID'].isin(target_employees)]
        else:
            target_df = self._identify_high_risk_employees('high_risk')

        n_targeted = len(target_df)
        if n_targeted == 0:
            raise ScenarioEngineError("No employees match the target criteria")

        affected_depts = target_df['Dept'].unique().tolist() if 'Dept' in target_df.columns else []
        avg_salary = target_df['Salary'].mean() if 'Salary' in target_df.columns else 75000

        # DATA-DRIVEN: Get baseline risk from actual predictions
        if '_ml_risk_score' in target_df.columns:
            baseline_risk = target_df['_ml_risk_score'].mean()
            risk_source = "ML predictions"
        elif 'Attrition' in self.df.columns:
            # Use historical attrition for similar employees
            baseline_risk = self._get_baseline_turnover(target_df)
            risk_source = "historical attrition"
        else:
            baseline_risk = 0.5
            risk_source = "estimate"

        # DATA-DRIVEN: Calculate effectiveness
        effectiveness, eff_method = self._calculate_intervention_effectiveness(
            intervention_type, target_df
        )

        # Calculate intervention cost
        if intervention_type == 'retention_bonus':
            bonus_pct = intervention_params.get('bonus_percentage', 10) / 100
            intervention_cost = n_targeted * avg_salary * bonus_pct
        elif intervention_type == 'career_path':
            per_person_cost = intervention_params.get('cost_per_person', 5000)
            intervention_cost = n_targeted * per_person_cost
        elif intervention_type == 'manager_change':
            intervention_cost = intervention_params.get('change_cost', 10000)
        else:
            intervention_cost = 10000

        # Calculate impact
        expected_retained = n_targeted * baseline_risk * effectiveness
        retention_improvement = effectiveness * baseline_risk

        replacement_savings = expected_retained * avg_salary * self.replacement_cost_mult

        cost_impact = CostImpact(
            salary_change=intervention_cost if intervention_type == 'retention_bonus' else 0,
            training_costs=intervention_cost if intervention_type == 'career_path' else 0,
            replacement_costs_avoided=replacement_savings,
            total_cost=intervention_cost,
            total_benefit=replacement_savings,
            net_impact=replacement_savings - intervention_cost
        )

        # Monte Carlo with data-driven uncertainty
        uncertainty = 0.15 if '_ml_risk_score' in target_df.columns else 0.25
        mc_result = self._run_monte_carlo(
            base_outcome=baseline_risk * (1 - effectiveness),
            outcome_std=baseline_risk * uncertainty,
            cost_per_outcome=avg_salary * self.replacement_cost_mult,
            n_affected=n_targeted,
            intervention_cost=intervention_cost
        )

        conf_level, conf_score = self._get_confidence_level(n_targeted)

        # Boost confidence if using ML predictions
        if '_ml_risk_score' in target_df.columns:
            conf_score = min(1.0, conf_score + 0.15)

        roi = safe_divide(cost_impact.net_impact, intervention_cost, 0) * 100

        # Build data-driven assumptions
        assumptions = [
            f"Target group baseline risk: {baseline_risk*100:.1f}% ({risk_source})",
            f"Intervention effectiveness: {effectiveness*100:.0f}% ({eff_method})",
            f"Replacement cost: {self.replacement_cost_mult}x salary"
        ]

        if '_ml_risk_score' in target_df.columns:
            assumptions.append(f"Targeting {n_targeted} employees identified by ML as high-risk")

        return ScenarioResult(
            scenario_id=scenario_id,
            scenario_name=f"{intervention_type.replace('_', ' ').title()} for {target_employees}",
            scenario_type='intervention',
            input_parameters={
                'intervention_type': intervention_type,
                'target_employees': target_employees,
                'params': intervention_params
            },
            affected_employees=n_targeted,
            affected_departments=affected_depts,
            baseline_turnover_rate=round(baseline_risk * 100, 1),
            projected_turnover_rate=round(baseline_risk * (1 - effectiveness) * 100, 1),
            turnover_change=round(retention_improvement * 100, 2),
            turnover_change_pct=round(effectiveness * 100, 1),
            simulation=mc_result,
            cost_impact=cost_impact,
            roi_estimate=round(roi, 1),
            payback_months=int(safe_divide(intervention_cost, replacement_savings / 12, 999)),
            confidence_level=conf_level,
            confidence_score=conf_score,
            assumptions=assumptions,
            risks=[
                "Individual response varies",
                "May not address root causes",
                "Could set precedent for future requests"
            ],
            recommendation="Recommended" if roi > 50 else "Evaluate alternatives",
            alternative_actions=[
                "Combination of interventions",
                "Target highest-risk individuals only",
                "Address systemic issues"
            ],
            computed_at=datetime.now().isoformat(),
            engines_used=self.available_engines,
            data_sources=self.data_sources
        )

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
            'recommended_scenario': best['scenario_name'],
            'reasoning': f"Highest expected ROI ({best['roi_estimate']}%) with "
                        f"{best['roi_positive_probability']*100:.0f}% probability of positive return"
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
