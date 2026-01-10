"""Analytics-related Pydantic schemas."""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel


class AnalyticsSummary(BaseModel):
    """Overall analytics summary."""
    headcount: int
    turnover_rate: Optional[float] = None
    department_count: int
    salary_mean: Optional[float] = None
    salary_median: Optional[float] = None
    tenure_mean: Optional[float] = None
    tenure_median: Optional[float] = None
    age_mean: Optional[float] = None
    lastrating_mean: Optional[float] = None
    attrition_count: Optional[int] = None
    active_count: Optional[int] = None
    takeaways: List[str] = []
    insights: Dict[str, str] = {}


class DepartmentStats(BaseModel):
    """Department-level statistics."""
    dept: str
    headcount: int
    avg_salary: Optional[float] = None
    median_salary: Optional[float] = None
    salary_std_dev: Optional[float] = None
    avg_tenure: Optional[float] = None
    avg_rating: Optional[float] = None
    avg_age: Optional[float] = None
    turnover_rate: Optional[float] = None


class DepartmentList(BaseModel):
    """List of department statistics."""
    departments: List[DepartmentStats]
    total_departments: int


class TenureDistribution(BaseModel):
    """Tenure distribution bucket."""
    tenure_range: str
    count: int
    turnover_rate: Optional[float] = None


class AgeDistribution(BaseModel):
    """Age distribution bucket."""
    age_range: str
    count: int


class SalaryBand(BaseModel):
    """Salary band information."""
    band: str
    lower: float
    upper: float
    count: int


class CorrelationData(BaseModel):
    """Feature correlation with target."""
    feature: str
    correlation: float
    abs_correlation: float


class HighRiskDepartment(BaseModel):
    """High-risk department information."""
    dept: str
    turnover_rate: float
    headcount: int
    avg_salary: Optional[float] = None
    avg_rating: Optional[float] = None
    reason: Optional[str] = None


class DistributionsResponse(BaseModel):
    """All distributions data."""
    tenure: List[TenureDistribution]
    age: List[AgeDistribution]
    salary_bands: List[SalaryBand]


class CorrelationsResponse(BaseModel):
    """Correlations with attrition."""
    correlations: List[CorrelationData]
    target_column: str


class HighRiskDepartmentsResponse(BaseModel):
    """High-risk departments response."""
    departments: List[HighRiskDepartment]
    threshold: float
