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
    pct_of_total: float
    avg_performance: Optional[float] = None
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
    quality_score: float
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
    recommendation: str


class HiringInsightsSummary(BaseModel):
    """Summary statistics for hiring insights."""
    total_employees: int
    sources_analyzed: int
    prehire_signals_available: int
    avg_performance: Optional[float] = None
    overall_retention: Optional[float] = None


class HiringInsights(BaseModel):
    """Strategic hiring insights."""
    summary: HiringInsightsSummary
    top_sources: List[SourceEffectiveness] = []
    top_predictors: List[PrehireCorrelation] = []
    red_flags: List[RedFlag] = []
    recommendations: List[str] = []
    roi_analysis: Dict[str, ROIAnalysis] = {}


class CohortPerformance(BaseModel):
    """Performance metrics for a cohort."""
    cohort_name: str
    count: int
    avg_performance: Optional[float] = None
    performance_std: Optional[float] = None
    high_performer_pct: Optional[float] = None
    low_performer_pct: Optional[float] = None
    retention_rate: Optional[float] = None
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
