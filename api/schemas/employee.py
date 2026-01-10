"""Employee-related Pydantic schemas."""

from typing import Optional, List
from pydantic import BaseModel, Field


class EmployeeBase(BaseModel):
    """Base employee schema with required fields."""
    employee_id: str = Field(..., alias="EmployeeID")
    dept: str = Field(..., alias="Dept")
    tenure: float = Field(..., alias="Tenure")
    salary: float = Field(..., alias="Salary")
    last_rating: float = Field(..., alias="LastRating")
    age: int = Field(..., alias="Age")

    class Config:
        populate_by_name = True


class EmployeeOptional(EmployeeBase):
    """Employee with optional fields."""
    attrition: Optional[int] = Field(None, alias="Attrition")
    performance_text: Optional[str] = Field(None, alias="PerformanceText")
    gender: Optional[str] = Field(None, alias="Gender")
    job_title: Optional[str] = Field(None, alias="JobTitle")

    class Config:
        populate_by_name = True


class RiskDriver(BaseModel):
    """Risk driver for an employee."""
    feature: str
    contribution: float
    value: Optional[float] = None
    abs_contribution: float


class EmployeeRisk(BaseModel):
    """Employee with risk prediction."""
    employee_id: str
    dept: str
    tenure: float
    salary: float
    last_rating: float
    risk_score: float
    risk_category: str
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    confidence_level: Optional[float] = None


class EmployeeDetail(BaseModel):
    """Detailed employee information with risk analysis."""
    employee_id: str
    dept: str
    tenure: float
    salary: float
    last_rating: float
    age: int
    risk_score: float
    risk_category: str
    risk_drivers: List[RiskDriver]
    recommendations: List[str]
    confidence: Optional[dict] = None


class EmployeeList(BaseModel):
    """List of employees."""
    employees: List[EmployeeRisk]
    total: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
