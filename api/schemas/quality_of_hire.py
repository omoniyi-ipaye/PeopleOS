"""
Pydantic schemas for Quality of Hire API endpoints.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class RiskFactorDetail(BaseModel):
    """Detailed risk factor information."""
    factor: str
    impact: str
    direction: str
    score: float
    description: str


class SourceEffectiveness(BaseModel):
    """Effectiveness metrics for a hiring source."""
    model_config = {'protected_namespaces': ()}
    HireSource: str
    hire_count: int
    total_hires: Optional[int] = None
    pct_of_total: float
    avg_performance: Optional[float] = None
    performance_recorded_observations: Optional[int] = None
    performance_observations: Optional[int] = None
    performance_coverage: Optional[float] = None
    performance_window_observations: Optional[int] = None
    performance_window_coverage: Optional[float] = None
    performance_maturity: Optional[str] = None
    promotion_observations: Optional[int] = None
    outcome_observations: Optional[int] = None
    retention_recorded_observations: Optional[int] = None
    retention_eligible_hires: Optional[int] = None
    retention_observations: Optional[int] = None
    retention_recorded_coverage: Optional[float] = None
    retention_coverage: Optional[float] = None
    retention_maturity: Optional[str] = None
    quality_components: List[str] = Field(default_factory=list)
    quality_weights: Dict[str, float] = Field(default_factory=dict)
    effective_quality_weights: Dict[str, float] = Field(default_factory=dict)
    configured_quality_weights: Dict[str, float] = Field(default_factory=dict)
    excluded_quality_components: List[str] = Field(default_factory=list)
    component_observations: Dict[str, int] = Field(default_factory=dict)
    component_coverage: Dict[str, float] = Field(default_factory=dict)
    minimum_component_observations: Optional[int] = None
    quality_unavailable_reason: Optional[str] = None
    quality_semantics: Optional[str] = None
    quality_claim: Optional[str] = None
    quality_comparison_status: Optional[str] = None
    quality_comparison_basis: Optional[str] = None
    role_mix: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    role_mix_columns: List[str] = Field(default_factory=list)
    high_performers: Optional[int] = None
    high_performer_rate: Optional[float] = None
    attrition_count: Optional[int] = None
    retention_rate: Optional[float] = None
    retention_rate_pct: Optional[float] = None
    avg_promotions: Optional[float] = None
    promoted_count: Optional[int] = None
    promotion_rate: Optional[float] = None
    avg_tenure: Optional[float] = None
    avg_interview_score: Optional[float] = None
    quality_score: Optional[float] = None
    grade: str  # A, B, C, D
    recommendation: str


class PrehireCorrelation(BaseModel):
    """Correlation between pre-hire signal and post-hire outcome."""
    predictor: str
    display_name: str
    correlation: float
    abs_correlation: float
    p_value: float
    is_significant: bool
    strength: str  # Strong, Moderate, Weak, Negligible
    direction: str  # positive or negative
    sample_size: int
    interpretation: str
    insight: Optional[str] = None


class CorrelationAnalysisResponse(BaseModel):
    """Pre-hire to post-hire correlation analysis."""
    available: bool
    reason: Optional[str] = None
    outcome_column: Optional[str] = None
    correlations: List[PrehireCorrelation] = []
    best_predictors: List[PrehireCorrelation] = []
    non_predictors: List[PrehireCorrelation] = []
    recommendations: List[str] = []
    outcome_observations: Optional[int] = None
    outcome_recorded_observations: Optional[int] = None
    outcome_maturity: Optional[str] = None
    measurement_gaps: List[Dict[str, Any]] = []


class RedFlag(BaseModel):
    """Red flag for hiring concern."""
    type: str
    source: str
    retention_rate: Optional[float] = None
    quality_score: Optional[float] = None
    message: str


class ROIAnalysis(BaseModel):
    """ROI analysis for a hiring source."""
    source: str
    quality_vs_average: float
    roi_indicator: str
    metric_semantics: str = 'relative_recorded_rating_not_return_on_investment'
    claim_boundary: str = 'descriptive_difference_not_hiring_effectiveness_or_financial_roi'
    recommendation: str


class HiringInsightsSummary(BaseModel):
    """Summary statistics for hiring insights."""
    total_employees: int
    sources_analyzed: int
    prehire_signals_available: int
    avg_performance: Optional[float] = None
    overall_retention: Optional[float] = None
    total_hires: Optional[int] = None
    performance_recorded_observations: Optional[int] = None
    performance_observations: Optional[int] = None
    performance_coverage: Optional[float] = None
    performance_window_coverage: Optional[float] = None
    retention_recorded_observations: Optional[int] = None
    retention_observations: Optional[int] = None
    retention_recorded_coverage: Optional[float] = None
    retention_coverage: Optional[float] = None
    performance_maturity: Optional[str] = None
    retention_maturity: Optional[str] = None
    performance_window_months: Optional[int] = None
    retention_window_months: Optional[int] = None
    role_mix_columns: List[str] = []


class HiringInsights(BaseModel):
    """Strategic hiring insights."""
    summary: HiringInsightsSummary
    top_sources: List[SourceEffectiveness] = []
    top_predictors: List[PrehireCorrelation] = []
    red_flags: List[RedFlag] = []
    recommendations: List[str] = []
    roi_analysis: Dict[str, ROIAnalysis] = {}


class CohortPerformance(BaseModel):
    performance_observations: Optional[int] = None
    outcome_observations: Optional[int] = None
    metric_semantics: str = 'observed_cohort_outcomes_not_exposure_adjusted_retention'
    """Performance metrics for a cohort."""
    cohort_name: str
    count: int
    total_hires: Optional[int] = None
    mature_hires: Optional[int] = None
    avg_performance: Optional[float] = None
    performance_std: Optional[float] = None
    performance_recorded_observations: Optional[int] = None
    performance_coverage: Optional[float] = None
    performance_maturity: Optional[str] = None
    high_performer_pct: Optional[float] = None
    low_performer_pct: Optional[float] = None
    retention_rate: Optional[float] = None
    retention_recorded_observations: Optional[int] = None
    retention_eligible_hires: Optional[int] = None
    retention_observations: Optional[int] = None
    retention_recorded_coverage: Optional[float] = None
    retention_coverage: Optional[float] = None
    retention_maturity: Optional[str] = None
    avg_tenure: Optional[float] = None


class NewHireRisk(BaseModel):
    """Risk assessment for a new hire."""
    EmployeeID: str
    HireDate: str
    HireSource: Optional[str] = None
    Dept: Optional[str] = None
    risk_score: float
    risk_category: str  # High, Medium, Low
    risk_factors: List[RiskFactorDetail] = []
    risk_factors_text: str = ""
    recommendation: str


class QualityOfHireSummary(BaseModel):
    """Summary of quality of hire analysis."""
    model_config = {'protected_namespaces': ()}
    total_employees: int
    has_hire_source: bool
    has_interview_scores: bool
    has_assessment: bool
    prehire_signals_count: int
    sources_analyzed: int
    best_source: Optional[str] = None
    top_predictor: Optional[str] = None
    new_hires_at_risk: int
    total_hires: Optional[int] = None
    best_source_semantics: Optional[str] = None
    performance_recorded_observations: Optional[int] = None
    performance_observations: Optional[int] = None
    performance_coverage: Optional[float] = None
    performance_window_coverage: Optional[float] = None
    retention_recorded_observations: Optional[int] = None
    retention_observations: Optional[int] = None
    retention_recorded_coverage: Optional[float] = None
    retention_coverage: Optional[float] = None
    performance_maturity: Optional[str] = None
    retention_maturity: Optional[str] = None
    performance_window_months: Optional[int] = None
    retention_window_months: Optional[int] = None
    role_mix_columns: List[str] = []


class QualityOfHireAnalysisResponse(BaseModel):
    """Full quality of hire analysis response."""
    model_config = {'protected_namespaces': ()}
    source_effectiveness: List[SourceEffectiveness] = []
    correlations: Optional[CorrelationAnalysisResponse] = None
    retention_correlations: Optional[CorrelationAnalysisResponse] = None
    insights: Optional[HiringInsights] = None
    cohort_analysis: List[CohortPerformance] = []
    new_hire_risks: List[NewHireRisk] = []
    summary: QualityOfHireSummary
    recommendations: List[str] = []
    warnings: List[str] = []
