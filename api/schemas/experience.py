"""
Pydantic schemas for Employee Experience API endpoints.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class ExperienceComponents(BaseModel):
    """Breakdown of EXI score components."""
    enps: Optional[float] = Field(None, description="eNPS contribution (0-100)")
    onboarding: Optional[float] = Field(None, description="Onboarding score contribution")
    pulse: Optional[float] = Field(None, description="Pulse survey contribution")
    manager: Optional[float] = Field(None, description="Manager satisfaction contribution")
    engagement: Optional[float] = Field(None, description="Engagement score contribution")
    work_life: Optional[float] = Field(None, description="Work-life balance contribution")
    career: Optional[float] = Field(None, description="Career growth contribution")
    derived: Optional[bool] = Field(None, description="True if EXI was derived from patterns")
    weights_used: Optional[Dict[str, float]] = Field(None, description="Weights applied")


class GroupExperience(BaseModel):
    """Experience metrics for a group (department, location, etc.)."""
    group: str
    exi: float = Field(..., description="Average EXI for group")
    count: int = Field(..., description="Number of employees in group")
    interpretation: str


class ExperienceIndexResponse(BaseModel):
    """Response for experience index calculation."""
    available: bool
    reason: Optional[str] = None
    overall_exi: Optional[float] = Field(None, description="Overall EXI score (0-100)")
    exi_std: Optional[float] = Field(None, description="Standard deviation of EXI")
    exi_median: Optional[float] = Field(None, description="Median EXI score")
    total_employees: Optional[int] = None
    signals_available: Optional[int] = Field(None, description="Number of experience signals")
    interpretation: Optional[str] = None
    benchmark: Optional[str] = Field(None, description="Above/Below average")
    by_group: Optional[List[GroupExperience]] = None


class EmployeeExperienceResponse(BaseModel):
    """Experience details for a single employee."""
    available: bool
    reason: Optional[str] = None
    EmployeeID: Optional[str] = None
    exi_score: Optional[float] = Field(None, description="Employee's EXI score (0-100)")
    segment: Optional[str] = Field(None, description="Engagement segment")
    interpretation: Optional[str] = None
    dept: Optional[str] = None
    components: Optional[ExperienceComponents] = None


class EngagementSegment(BaseModel):
    """Engagement segment with statistics."""
    segment: str = Field(..., description="Segment name (Thriving, Content, etc.)")
    count: int = Field(..., description="Number of employees")
    percentage: float = Field(..., description="Percentage of workforce")
    avg_exi: float = Field(..., description="Average EXI in segment")
    exi_range: str = Field(..., description="EXI score range for segment")


class SegmentsResponse(BaseModel):
    """Response for engagement segmentation."""
    available: bool
    reason: Optional[str] = None
    segments: Optional[List[EngagementSegment]] = None
    total_employees: Optional[int] = None
    health_indicator: Optional[str] = Field(None, description="Healthy, Moderate, At Risk")
    thriving_percentage: Optional[float] = None
    at_risk_percentage: Optional[float] = None
    recommendations: Optional[List[str]] = None


class ExperienceDriver(BaseModel):
    """Factor that drives experience scores."""
    factor: str = Field(..., description="Factor/column name")
    correlation: float = Field(..., description="Correlation with EXI")
    impact: str = Field(..., description="High, Medium, or Low")
    direction: str = Field(..., description="Positive or Negative")


class DriversResponse(BaseModel):
    """Response for experience drivers analysis."""
    available: bool
    reason: Optional[str] = None
    drivers: Optional[List[ExperienceDriver]] = None
    top_positive_drivers: Optional[List[ExperienceDriver]] = None
    top_negative_drivers: Optional[List[ExperienceDriver]] = None
    recommendations: Optional[List[str]] = None


class AtRiskExperience(BaseModel):
    """Employee at risk due to low experience."""
    EmployeeID: str
    Dept: Optional[str] = None
    current_exi: float = Field(..., description="Current EXI score")
    segment: str = Field(..., description="Current segment")
    tenure: Optional[float] = None
    risk_factors: List[str] = Field(default_factory=list)
    recommended_actions: List[str] = Field(default_factory=list)


class AtRiskByDepartment(BaseModel):
    """At-risk count by department."""
    department: str
    at_risk_count: int


class AtRiskResponse(BaseModel):
    """Response for at-risk employees."""
    available: bool
    reason: Optional[str] = None
    total_at_risk: Optional[int] = None
    threshold_used: Optional[float] = None
    employees: Optional[List[AtRiskExperience]] = None
    by_department: Optional[List[AtRiskByDepartment]] = None


class LifecycleStage(BaseModel):
    """Experience metrics for a lifecycle stage."""
    stage: str = Field(..., description="New Hire, Ramping, Established, Veteran")
    count: int
    avg_exi: float
    at_risk_count: int


class LifecycleResponse(BaseModel):
    """Response for lifecycle experience analysis."""
    available: bool
    reason: Optional[str] = None
    stages: Optional[List[LifecycleStage]] = None
    concerns: Optional[List[str]] = None
    recommendations: Optional[List[str]] = None


class ManagerStats(BaseModel):
    """Manager's team experience statistics."""
    ManagerID: str
    team_size: int
    avg_team_exi: float
    at_risk_count: int
    at_risk_percentage: float


class ManagerImpactResponse(BaseModel):
    """Response for manager impact analysis."""
    available: bool
    reason: Optional[str] = None
    managers_analyzed: Optional[int] = None
    overall_avg_exi: Optional[float] = None
    managers_below_average: Optional[int] = None
    bottom_managers: Optional[List[ManagerStats]] = None
    top_managers: Optional[List[ManagerStats]] = None
    recommendations: Optional[List[str]] = None


class SignalsResponse(BaseModel):
    """Response for available experience signals."""
    has_enps: bool = False
    has_onboarding: bool = False
    has_pulse: bool = False
    has_manager_satisfaction: bool = False
    has_engagement: bool = False
    has_work_life: bool = False
    has_career_growth: bool = False
    total_signals: int = 0
    coverage_percentage: float = Field(0, description="% employees with any signal")
    recommendations: List[str] = Field(default_factory=list)


class ExperienceSummary(BaseModel):
    """Summary of experience analysis."""
    overall_exi: Optional[float] = None
    health_indicator: str = "Unknown"
    total_employees: int = 0
    at_risk_count: int = 0
    signals_available: int = 0
    total_warnings: int = 0
    total_recommendations: int = 0


class ExperienceAnalysisResponse(BaseModel):
    """Full experience analysis response."""
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
