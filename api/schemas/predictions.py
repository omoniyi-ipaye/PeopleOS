"""Prediction-related Pydantic schemas."""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel


class ModelMetrics(BaseModel):
    """ML model performance metrics."""
    model_config = {'protected_namespaces': ()}
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: Optional[float] = None
    best_model: str
    train_size: int
    test_size: int
    reliability: str
    warnings: Optional[List[str]] = None


class FeatureImportance(BaseModel):
    """Feature importance ranking."""
    feature: str
    importance: float


class FeatureImportanceResponse(BaseModel):
    """Feature importance response."""
    features: List[FeatureImportance]
    model_name: str


class RiskPrediction(BaseModel):
    """Risk prediction for an employee."""
    employee_id: str
    risk_score: float
    risk_category: str
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None


class RiskDistribution(BaseModel):
    """Risk distribution summary."""
    high_risk: int
    medium_risk: int
    low_risk: int
    total: int
    high_risk_pct: float
    medium_risk_pct: float
    low_risk_pct: float


class EmployeeRiskDetail(BaseModel):
    """Detailed risk analysis for individual employee."""
    employee_id: str
    dept: str
    tenure: float
    salary: float
    last_rating: float
    age: int
    risk_score: float
    risk_category: str
    drivers: List[Dict[str, Any]]
    recommendations: List[str]
    base_value: Optional[float] = None
    confidence: Optional[Dict[str, Any]] = None


class PredictionsResponse(BaseModel):
    """Full predictions response."""
    model_config = {'protected_namespaces': ()}
    predictions: List[RiskPrediction]
    distribution: RiskDistribution
    model_metrics: ModelMetrics
