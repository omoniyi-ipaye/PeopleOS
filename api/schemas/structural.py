"""
Pydantic schemas for Structural Analysis API endpoints.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class StagnationEmployee(BaseModel):
    """Stagnation metrics for an individual employee."""
    EmployeeID: str
    Tenure: float
    YearsInCurrentRole: float
    StagnationIndex: float
    StagnationCategory: str  # Critical, Warning, Monitor, Healthy, Too Early
    Dept: Optional[str] = None
    JobTitle: Optional[str] = None
    JobLevel: Optional[int] = None
    LastRating: Optional[float] = None
    Location: Optional[str] = None
    YearsSinceLastPromotion: Optional[float] = None


class StagnationHotspot(BaseModel):
    """Stagnation hotspot in the organization."""
    type: str  # department, job_level
    name: str
    avg_stagnation_index: float
    employees_at_risk: int
    total_employees: int
    risk_pct: float
    severity: str  # Critical, Warning


class StagnationSummary(BaseModel):
    """Summary of stagnation analysis."""
    total_analyzed: int
    critical_count: int
    warning_count: int
    avg_stagnation_index: float


class StagnationHotspotsResponse(BaseModel):
    """Response for stagnation hotspots analysis."""
    available: bool
    reason: Optional[str] = None
    hotspots: List[StagnationHotspot] = []
    critical_employees: List[Dict[str, Any]] = []
    summary: Optional[StagnationSummary] = None


class ManagerSpan(BaseModel):
    """Span of control metrics for a manager."""
    ManagerID: str
    DirectReports: int
    SpanCategory: str  # Under-Leveraged, Optimal, Stretched, Overloaded, Critical
    BurnoutRiskScore: int  # 0-100
    Dept: Optional[str] = None
    JobTitle: Optional[str] = None
    JobLevel: Optional[int] = None
    Location: Optional[str] = None
    LastRating: Optional[float] = None
    Tenure: Optional[float] = None


class DepartmentSpanSummary(BaseModel):
    """Span of control summary by department."""
    department: str
    total_managers: int
    at_risk_count: int
    avg_span: float
    max_span: int


class SpanOfControlSummary(BaseModel):
    """Summary of span of control analysis."""
    total_managers: int
    at_risk_count: int
    avg_span: float
    max_span: int
    optimal_count: int
    under_leveraged_count: int


class SpanOfControlResponse(BaseModel):
    """Response for span of control analysis."""
    available: bool
    reason: Optional[str] = None
    at_risk_managers: List[Dict[str, Any]] = []
    department_summary: List[DepartmentSpanSummary] = []
    summary: Optional[SpanOfControlSummary] = None
    recommendations: List[str] = []


class GroupStatistic(BaseModel):
    """Statistics for a demographic group."""
    group: str
    mean_years: float
    std_years: Optional[float] = None
    count: int


class PromotionGap(BaseModel):
    """Promotion gap for a group vs reference."""
    group: str
    mean_years: float
    gap_vs_reference: float
    count: int


class EquityAuditResult(BaseModel):
    """Equity audit result for a single attribute."""
    attribute: str
    reference_group: str
    group_statistics: List[Dict[str, Any]]
    gaps: List[PromotionGap]
    p_value: float
    significant_gap: bool
    finding: str


class EquityAuditSummary(BaseModel):
    """Summary of equity audit."""
    employees_analyzed: int
    attributes_tested: int
    significant_gaps_found: int


class PromotionEquityResponse(BaseModel):
    """Response for promotion velocity equity audit."""
    available: bool
    reason: Optional[str] = None
    audit_results: List[EquityAuditResult] = []
    significant_gaps: List[EquityAuditResult] = []
    summary: Optional[EquityAuditSummary] = None
    recommendations: List[str] = []
    methodology: Optional[str] = None


class PromotionBottleneck(BaseModel):
    """Promotion bottleneck in the organization."""
    type: str  # department, job_level
    name: str
    avg_years_since_promotion: float
    median_years: float
    employee_count: int
    above_average_by: Optional[float] = None  # Percentage above org average


class BottleneckEmployee(BaseModel):
    """Employee waiting longest for promotion."""
    EmployeeID: str
    Dept: Optional[str] = None
    JobLevel: Optional[int] = None
    JobTitle: Optional[str] = None
    YearsSinceLastPromotion: float
    LastRating: Optional[float] = None
    Tenure: Optional[float] = None


class BottleneckSummary(BaseModel):
    """Summary of promotion bottleneck analysis."""
    avg_years_since_promotion: float
    median_years: float
    employees_over_5_years: int


class PromotionBottlenecksResponse(BaseModel):
    """Response for promotion bottleneck analysis."""
    available: bool
    reason: Optional[str] = None
    bottlenecks: List[PromotionBottleneck] = []
    employees_waiting_longest: List[Dict[str, Any]] = []
    summary: Optional[BottleneckSummary] = None


class StructuralAnalysisSummary(BaseModel):
    """Summary of all structural analyses."""
    stagnation_analysis: bool
    span_of_control: bool
    promotion_equity: bool
    total_warnings: int
    total_recommendations: int


class StructuralAnalysisResponse(BaseModel):
    """Complete structural analysis response."""
    stagnation: Dict[str, Any]
    span_of_control: Dict[str, Any]
    promotion_equity: Dict[str, Any]
    promotion_bottlenecks: Dict[str, Any]
    summary: StructuralAnalysisSummary
    recommendations: List[str] = []
    warnings: List[str] = []
