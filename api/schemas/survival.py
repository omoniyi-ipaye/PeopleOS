"""
Pydantic schemas for Survival Analysis API endpoints.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union


class SurvivalPoint(BaseModel):
    """Single point on survival curve."""
    model_config = {'protected_namespaces': ()}
    time_months: float
    time_years: float
    survival_probability: float
    at_risk: Optional[int] = None


class KaplanMeierResult(BaseModel):
    """Kaplan-Meier survival curve result."""
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
    """Survival metrics for a segment (department, location, etc.)."""
    segment_name: str
    median_survival_months: Optional[float] = None
    sample_size: int
    events: int
    survival_function: Optional[List[Dict[str, float]]] = None


class CoxCoefficient(BaseModel):
    """Cox model coefficient with interpretation."""
    feature: str
    coefficient: float
    hazard_ratio: float
    p_value: float
    is_significant: bool
    ci_lower: float
    ci_upper: float
    direction: str  # 'increases' or 'decreases'
    interpretation: str


class CoxModelMetrics(BaseModel):
    """Cox model performance metrics."""
    concordance_index: float
    log_likelihood: float
    aic: float
    sample_size: int
    events: int
    quality_interpretation: str


class CoxModelResult(BaseModel):
    """Full Cox Proportional Hazards model result."""
    model_config = {'protected_namespaces': ()}
    available: bool
    reason: Optional[str] = None
    coefficients: Optional[Dict[str, CoxCoefficient]] = None
    model_metrics: Optional[CoxModelMetrics] = None
    covariates_used: Optional[List[str]] = None
    recommendations: Optional[List[str]] = None


class CohortInsight(BaseModel):
    """Insight for a specific employee cohort."""
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
    """Single point on hazard function."""
    time_years: float
    baseline_hazard: float
    cumulative_hazard: float
    survival: float


class RiskPeriod(BaseModel):
    """Period of elevated attrition risk."""
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
    """Detailed risk factor for an individual employee."""
    model_config = {'protected_namespaces': ()}
    factor: str
    impact: str  # 'High', 'Medium', 'Low'
    direction: str # 'Increase Risk', 'Decrease Risk'
    score: float
    description: str


class AtRiskEmployee(BaseModel):
    """Employee at risk of attrition."""
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
    """Summary of survival analysis."""
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
    """Full survival analysis response."""
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
    """Survival prediction for a single employee."""
    EmployeeID: str
    survival_3mo: float
    survival_6mo: float
    survival_12mo: float
    survival_24mo: Optional[float] = None
    attrition_risk_12mo: float
    risk_category: str
    current_tenure_years: float
    risk_factors: Optional[List[RiskFactor]] = None
