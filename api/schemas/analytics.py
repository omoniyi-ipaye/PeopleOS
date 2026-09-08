"""Analytics API schemas with explicit metric semantics."""

from typing import Optional, List, Dict
from pydantic import BaseModel


class AnalyticsSummary(BaseModel):
    """Current-state analytics summary."""
    headcount: int
    record_count: Optional[int] = None
    observed_attrition_share: Optional[float] = None
    # Backward-compatible alias. Semantics are observed attrition share, not period turnover.
    turnover_rate: Optional[float] = None
    turnover_rate_semantics: Optional[str] = None
    department_count: int
    salary_mean: Optional[float] = None
    salary_median: Optional[float] = None
    tenure_mean: Optional[float] = None
    tenure_median: Optional[float] = None
    age_mean: Optional[float] = None
    lastrating_mean: Optional[float] = None
    attrition_count: Optional[int] = None
    active_count: Optional[int] = None
    attrition_known_count: Optional[int] = None
    salary_observations: Optional[int] = None
    salary_excluded_count: Optional[int] = None
    tenure_observations: Optional[int] = None
    tenure_excluded_count: Optional[int] = None
    age_observations: Optional[int] = None
    age_excluded_count: Optional[int] = None
    lastrating_observations: Optional[int] = None
    lastrating_excluded_count: Optional[int] = None
    takeaways: List[str] = []
    insights: Dict[str, str] = {}


class DepartmentStats(BaseModel):
    outcome_observations: int = 0
    dept: str
    headcount: int
    total_records: Optional[int] = None
    avg_salary: Optional[float] = None
    median_salary: Optional[float] = None
    salary_std_dev: Optional[float] = None
    avg_tenure: Optional[float] = None
    avg_rating: Optional[float] = None
    avg_age: Optional[float] = None
    observed_attrition_share: Optional[float] = None
    turnover_rate: Optional[float] = None  # compatibility alias


class DepartmentList(BaseModel):
    departments: List[DepartmentStats]
    total_departments: int
    minimum_group_size: int = 10
    suppressed_department_count: int = 0


class TenureDistribution(BaseModel):
    tenure_range: str
    count: int
    observed_attrition_share: Optional[float] = None
    turnover_rate: Optional[float] = None  # compatibility alias


class AgeDistribution(BaseModel):
    age_range: str
    count: int


class SalaryBand(BaseModel):
    band: str
    lower: float
    upper: float
    count: int


class CorrelationData(BaseModel):
    """Observed association with target; not a causal driver."""
    feature: str
    correlation: float
    abs_correlation: float


class HighRiskDepartment(BaseModel):
    """Department above configured observed-attrition-share screening threshold."""
    dept: str
    observed_attrition_share: float
    turnover_rate: float  # compatibility alias
    headcount: int
    avg_salary: Optional[float] = None
    avg_rating: Optional[float] = None
    reason: Optional[str] = None


class DistributionsResponse(BaseModel):
    tenure: List[TenureDistribution]
    age: List[AgeDistribution]
    salary_bands: List[SalaryBand]


class CorrelationsResponse(BaseModel):
    correlations: List[CorrelationData]
    target_column: str
    metric_semantics: str = "observational_association_not_causal_effect"


class HighRiskDepartmentsResponse(BaseModel):
    departments: List[HighRiskDepartment]
    threshold: float
    minimum_group_size: int = 10
    suppressed_department_count: int = 0
    threshold_semantics: str = "observed_attrition_share_screening_threshold"
