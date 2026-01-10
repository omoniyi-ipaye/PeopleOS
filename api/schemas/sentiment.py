"""
Pydantic schemas for Sentiment Analysis API endpoints.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class ENPSGroupResult(BaseModel):
    """eNPS result for a group/cohort."""
    group: str
    enps: Optional[float] = None
    responses: int
    promoters: int
    detractors: int


class ENPSResponse(BaseModel):
    """eNPS calculation response."""
    available: bool
    reason: Optional[str] = None
    overall_enps: Optional[float] = None
    total_responses: Optional[int] = None
    promoters: Optional[int] = None
    passives: Optional[int] = None
    detractors: Optional[int] = None
    promoter_pct: Optional[float] = None
    passive_pct: Optional[float] = None
    detractor_pct: Optional[float] = None
    interpretation: Optional[str] = None
    by_group: List[ENPSGroupResult] = []


class ENPSTrendPoint(BaseModel):
    """Single point in eNPS trend."""
    period: str
    enps: Optional[float] = None
    responses: int
    promoters: int
    detractors: int


class ENPSTrendsResponse(BaseModel):
    """eNPS trends over time response."""
    available: bool
    reason: Optional[str] = None
    period_type: Optional[str] = None
    trends: List[ENPSTrendPoint] = []
    trend_direction: Optional[str] = None
    recent_change: Optional[float] = None


class ENPSDriver(BaseModel):
    """Driver analysis for eNPS scores."""
    dimension: str
    correlation: Optional[float] = None
    avg_score: Optional[float] = None
    impact: str  # High, Medium, Low


class ENPSDriversResponse(BaseModel):
    """eNPS drivers analysis response."""
    available: bool
    reason: Optional[str] = None
    drivers: List[ENPSDriver] = []
    top_driver: Optional[str] = None
    improvement_areas: List[str] = []
    recommendations: List[str] = []


class OnboardingTrajectory(BaseModel):
    """Onboarding trajectory for an employee."""
    EmployeeID: str
    Dept: Optional[str] = None
    scores: Dict[str, float] = {}  # {"30-day": 4.2, "60-day": 3.8, ...}
    trajectory_direction: str  # improving, declining, stable, insufficient_data
    trend_change: Optional[float] = None
    surveys_completed: int
    latest_score: Optional[float] = None
    at_risk: bool


class OnboardingTrajectorySummary(BaseModel):
    """Summary of onboarding trajectory analysis."""
    total_employees: int
    declining_count: int
    at_risk_count: int
    improving_count: int
    avg_completion_rate: float


class OnboardingTrajectoryResponse(BaseModel):
    """Onboarding trajectory analysis response."""
    available: bool
    reason: Optional[str] = None
    trajectories: List[OnboardingTrajectory] = []
    summary: Optional[OnboardingTrajectorySummary] = None
    at_risk_employees: List[OnboardingTrajectory] = []


class DimensionScore(BaseModel):
    """Score for an onboarding dimension."""
    dimension: str
    avg_score: Optional[float] = None


class SurveyTypeMetrics(BaseModel):
    """Metrics for a survey type (30/60/90 day)."""
    survey_type: str
    avg_score: float
    responses: int
    healthy_pct: float


class OnboardingHealthResponse(BaseModel):
    """Onboarding health assessment response."""
    available: bool
    reason: Optional[str] = None
    by_survey_type: List[SurveyTypeMetrics] = []
    dimension_scores: List[DimensionScore] = []
    weakest_dimensions: List[DimensionScore] = []
    overall_health: Optional[str] = None  # Healthy, Needs Attention
    recommendations: List[str] = []


class EarlyWarning(BaseModel):
    """Early warning for an at-risk employee."""
    EmployeeID: str
    Dept: Optional[str] = None
    warning_type: str  # eNPS Detractor, Declining Onboarding
    severity: str  # High, Medium, Low
    details: str


class EarlyWarningSummary(BaseModel):
    """Summary of early warnings."""
    total_at_risk: int
    high_severity: int
    medium_severity: int
    warning_types: List[str]


class EarlyWarningsResponse(BaseModel):
    """Early warnings detection response."""
    available: bool
    warnings: List[EarlyWarning] = []
    summary: Optional[EarlyWarningSummary] = None
    recommendations: List[str] = []


class SentimentSummary(BaseModel):
    """Summary of all sentiment analyses."""
    enps_available: bool
    onboarding_available: bool
    overall_enps: Optional[float] = None
    employees_at_risk: int
    total_warnings: int
    total_recommendations: int


class SentimentAnalysisResponse(BaseModel):
    """Complete sentiment analysis response."""
    enps: Dict[str, Any]
    enps_trends: Dict[str, Any]
    enps_drivers: Dict[str, Any]
    onboarding: Dict[str, Any]
    onboarding_health: Dict[str, Any]
    early_warnings: Dict[str, Any]
    summary: SentimentSummary
    recommendations: List[str] = []
    warnings: List[str] = []


class SurveyUploadResponse(BaseModel):
    """Response after uploading survey data."""
    success: bool
    message: str
    rows_loaded: int
    survey_type: str
    columns_found: List[str]
    warnings: List[str] = []
