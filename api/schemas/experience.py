"""Pydantic schemas for governed Employee Experience API endpoints."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ExperienceComponents(BaseModel):
    enps: Optional[float] = None
    onboarding: Optional[float] = None
    pulse: Optional[float] = None
    manager: Optional[float] = None
    engagement: Optional[float] = None
    work_life: Optional[float] = None
    career: Optional[float] = None
    derived: Optional[bool] = None
    weights_used: Optional[Dict[str, float]] = None


class GroupExperience(BaseModel):
    group: str
    exi: float
    count: int
    interpretation: str


class ExperienceIndexResponse(BaseModel):
    available: bool
    reason: Optional[str] = None
    overall_exi: Optional[float] = None
    exi_std: Optional[float] = None
    exi_median: Optional[float] = None
    total_employees: Optional[int] = None
    respondent_count: Optional[int] = None
    response_coverage: Optional[float] = None
    signals_available: Optional[int] = None
    interpretation: Optional[str] = None
    benchmark: Optional[str] = None
    by_group: Optional[List[GroupExperience]] = None


class EmployeeExperienceResponse(BaseModel):
    available: bool
    reason: Optional[str] = None
    EmployeeID: Optional[str] = None
    exi_score: Optional[float] = None
    segment: Optional[str] = None
    interpretation: Optional[str] = None
    dept: Optional[str] = None
    components: Optional[ExperienceComponents] = None


class EngagementSegment(BaseModel):
    """Configured score band; suppressed cells carry no reconstructable value."""
    segment: str
    count: Optional[int] = None
    percentage: Optional[float] = None
    avg_exi: Optional[float] = None
    exi_range: str
    suppressed: bool = False


class SegmentsResponse(BaseModel):
    available: bool
    reason: Optional[str] = None
    segments: Optional[List[EngagementSegment]] = None
    total_employees: Optional[int] = None
    health_indicator: Optional[str] = None
    thriving_percentage: Optional[float] = None
    at_risk_percentage: Optional[float] = None
    recommendations: Optional[List[str]] = None
    suppression_applied: bool = False


class ExperienceDriver(BaseModel):
    sample_size: int = 0
    metric_semantics: str = "observational_association_excluding_index_components"
    factor: str
    correlation: float
    impact: str
    direction: str


class DriversResponse(BaseModel):
    available: bool
    reason: Optional[str] = None
    drivers: Optional[List[ExperienceDriver]] = None
    top_positive_drivers: Optional[List[ExperienceDriver]] = None
    top_negative_drivers: Optional[List[ExperienceDriver]] = None
    recommendations: Optional[List[str]] = None


class AtRiskExperience(BaseModel):
    EmployeeID: str
    Dept: Optional[str] = None
    current_exi: float
    segment: str
    tenure: Optional[float] = None
    risk_factors: List[str] = Field(default_factory=list)
    recommended_actions: List[str] = Field(default_factory=list)


class AtRiskByDepartment(BaseModel):
    department: str
    at_risk_count: int


class AtRiskResponse(BaseModel):
    available: bool
    reason: Optional[str] = None
    total_at_risk: Optional[int] = None
    threshold_used: Optional[float] = None
    employees: Optional[List[AtRiskExperience]] = None
    by_department: Optional[List[AtRiskByDepartment]] = None
    suppressed: bool = False
    metric_semantics: Optional[str] = None


class LifecycleStage(BaseModel):
    stage: str
    count: int
    avg_exi: Optional[float] = None
    respondent_count: int = 0
    at_risk_count: Optional[int] = None
    at_risk_suppressed: bool = False


class LifecycleResponse(BaseModel):
    available: bool
    reason: Optional[str] = None
    stages: Optional[List[LifecycleStage]] = None
    concerns: Optional[List[str]] = None
    recommendations: Optional[List[str]] = None


class ManagerStats(BaseModel):
    ManagerID: str
    team_size: int
    avg_team_exi: float
    at_risk_count: int
    at_risk_percentage: float


class ManagerImpactResponse(BaseModel):
    available: bool
    reason: Optional[str] = None
    managers_analyzed: Optional[int] = None
    overall_avg_exi: Optional[float] = None
    managers_below_average: Optional[int] = None
    bottom_managers: Optional[List[ManagerStats]] = None
    top_managers: Optional[List[ManagerStats]] = None
    recommendations: Optional[List[str]] = None


class SignalsResponse(BaseModel):
    has_enps: bool = False
    has_onboarding: bool = False
    has_pulse: bool = False
    has_manager_satisfaction: bool = False
    has_engagement: bool = False
    has_work_life: bool = False
    has_career_growth: bool = False
    total_signals: int = 0
    coverage_percentage: float = 0
    recommendations: List[str] = Field(default_factory=list)


class ExperienceSummary(BaseModel):
    overall_exi: Optional[float] = None
    health_indicator: str = "Unknown"
    total_employees: int = 0
    at_risk_count: Optional[int] = None
    signals_available: int = 0
    total_warnings: int = 0
    total_recommendations: int = 0


class ExperienceAnalysisResponse(BaseModel):
    experience_index: ExperienceIndexResponse
    segments: SegmentsResponse
    drivers: DriversResponse
    at_risk: AtRiskResponse
    lifecycle: LifecycleResponse
    manager_impact: ManagerImpactResponse
    signals: SignalsResponse
    summary: ExperienceSummary
    recommendations: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
