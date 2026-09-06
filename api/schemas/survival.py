"""Pydantic schemas for governed cohort-level Survival Analysis API endpoints."""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class SurvivalPoint(BaseModel):
    """Single point on a cohort survival curve."""
    model_config = {'protected_namespaces': ()}
    time_months: float
    time_years: float
    survival_probability: float
    at_risk: Optional[int] = None


class KaplanMeierResult(BaseModel):
    """Kaplan-Meier cohort survival result."""
    model_config = {'protected_namespaces': ()}
    survival_function: List[SurvivalPoint]
    median_survival_months: Optional[float] = None
    median_survival_years: Optional[float] = None
    mean_survival_months: Optional[float] = None
    survival_at_6mo: Optional[float] = None
    survival_at_12mo: Optional[float] = None
    survival_at_24mo: Optional[float] = None
    survival_at_36mo: Optional[float] = None
    survival_at_60mo: Optional[float] = None


class SegmentSurvival(BaseModel):
    """Survival metrics for an aggregate segment."""
    segment_name: str
    median_survival_months: Optional[float] = None
    sample_size: int
    events: int
    survival_function: Optional[List[Dict[str, float]]] = None


class CoxCoefficient(BaseModel):
    """Cox association estimate.

    Statistical estimates and confidence bounds are nullable because sparse or
    separated data can make an estimate undefined or unbounded. Null is the
    truthful representation; PeopleOS must never substitute a fabricated zero.
    """
    feature: str
    coefficient: Optional[float] = None
    hazard_ratio: Optional[float] = None
    p_value: Optional[float] = None
    is_significant: bool = False
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    direction: str
    interpretation: str


class CoxModelMetrics(BaseModel):
    """Cox model fitness metrics; undefined estimates remain null."""
    concordance_index: Optional[float] = None
    log_likelihood: Optional[float] = None
    aic: Optional[float] = None
    sample_size: int
    events: int
    quality_interpretation: str


class CoxModelResult(BaseModel):
    """Full Cox Proportional Hazards association result."""
    model_config = {'protected_namespaces': ()}
    available: bool
    reason: Optional[str] = None
    coefficients: Optional[Dict[str, CoxCoefficient]] = None
    model_metrics: Optional[CoxModelMetrics] = None
    covariates_used: Optional[List[str]] = None
    recommendations: Optional[List[str]] = None


class CohortInsight(BaseModel):
    """Insight for a workforce cohort."""
    cohort_description: str
    cohort_size: int
    filters_applied: Dict[str, Any]
    attrition_count: Optional[int] = None
    attrition_rate: Optional[float] = None
    avg_tenure_years: Optional[float] = None
    survival_probability_3mo: Optional[float] = None
    survival_probability_6mo: Optional[float] = None
    survival_probability_12mo: Optional[float] = None
    median_survival_months: Optional[float] = None
    median_survival_years: Optional[float] = None
    key_risk_factors: Optional[List[str]] = None
    narrative: Optional[str] = None
    warning: Optional[str] = None


class HazardPoint(BaseModel):
    """Single point on the baseline hazard function."""
    time_years: float
    baseline_hazard: float
    cumulative_hazard: float
    survival: float


class RiskPeriod(BaseModel):
    """Period of elevated cohort hazard."""
    time_years: float
    relative_risk: float
    interpretation: str


class HazardOverTime(BaseModel):
    """Hazard function over time analysis."""
    available: bool
    reason: Optional[str] = None
    hazard_over_time: Optional[List[HazardPoint]] = None
    risk_periods: Optional[List[RiskPeriod]] = None


class RiskFactor(BaseModel):
    """Legacy individual risk-factor schema retained for compatibility only."""
    model_config = {'protected_namespaces': ()}
    factor: str
    impact: str
    direction: str
    score: float
    description: str


class AtRiskEmployee(BaseModel):
    """Legacy schema retained for compatibility; enterprise endpoints return none."""
    EmployeeID: str
    survival_3mo: Optional[float] = None
    survival_6mo: Optional[float] = None
    survival_12mo: Optional[float] = None
    attrition_risk_3mo: Optional[float] = None
    attrition_risk_6mo: Optional[float] = None
    attrition_risk_12mo: Optional[float] = None
    risk_category: str
    current_tenure_years: Optional[float] = None
    current_rating: Optional[float] = None
    Dept: Optional[str] = None
    Location: Optional[str] = None
    JobTitle: Optional[str] = None
    YearsSinceLastPromotion: Optional[float] = None
    CompaRatio: Optional[float] = None
    risk_factors: Optional[List[RiskFactor]] = None


class SurvivalSummary(BaseModel):
    """Cohort survival analysis summary."""
    total_employees: int
    attrition_available: bool
    attrition_count: Optional[int] = None
    overall_attrition_rate: Optional[float] = None
    cox_model_fitted: bool
    covariates_used: List[str]
    high_risk_count: int
    medium_risk_count: int
    median_tenure: Optional[float] = None
    avg_12mo_risk: Optional[float] = None


class SurvivalAnalysisResponse(BaseModel):
    """Full governed cohort-level survival analysis response."""
    model_config = {'protected_namespaces': ()}
    kaplan_meier: Optional[Dict[str, Any]] = None
    kaplan_meier_by_dept: Optional[Dict[str, Any]] = None
    cox_model: Optional[CoxModelResult] = None
    hazard_over_time: Optional[HazardOverTime] = None
    cohort_insights: List[CohortInsight] = []
    at_risk_employees: List[AtRiskEmployee] = []
    summary: SurvivalSummary
    recommendations: List[str] = []
    warnings: List[str] = []


class EmployeeSurvivalPrediction(BaseModel):
    """Legacy individual schema retained for compatibility; route is disabled."""
    EmployeeID: str
    survival_3mo: float
    survival_6mo: float
    survival_12mo: float
    survival_24mo: Optional[float] = None
    attrition_risk_12mo: float
    risk_category: str
    current_tenure_years: float
    risk_factors: Optional[List[RiskFactor]] = None
