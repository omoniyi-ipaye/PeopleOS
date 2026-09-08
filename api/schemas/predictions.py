"""Prediction-related API schemas."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class ModelMetrics(BaseModel):
    model_config = {'protected_namespaces': ()}
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: Optional[float] = None
    brier_score: Optional[float] = None
    calibration_error: Optional[float] = None
    baseline_brier_score: Optional[float] = None
    brier_skill_score: Optional[float] = None
    average_precision: Optional[float] = None
    baseline_average_precision: Optional[float] = None
    validation_checks: Optional[Dict[str, bool]] = None
    future_departure_validated: bool = False
    best_model: str
    train_size: int
    test_size: int
    reliability: str
    warnings: Optional[List[str]] = None
    evaluation_semantics: str = 'held_out_evaluation_after_training_only_preprocessing'


class FeatureImportance(BaseModel):
    feature: str
    importance: float


class FeatureImportanceResponse(BaseModel):
    features: List[FeatureImportance]
    model_name: str
    interpretation: str = 'Feature importance describes model influence and does not establish causation.'


class RiskPrediction(BaseModel):
    """Deprecated individual-risk shape retained only for response compatibility."""
    employee_id: str
    risk_score: float
    risk_category: str
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None


class RiskDistribution(BaseModel):
    high_risk: int
    medium_risk: int
    low_risk: int
    total: int
    high_risk_pct: float
    medium_risk_pct: float
    low_risk_pct: float


class EmployeeRiskDetail(BaseModel):
    """Legacy shape; individual predictive views are disabled by policy."""
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
    model_config = {'protected_namespaces': ()}
    predictions: List[RiskPrediction] = []
    distribution: RiskDistribution
    model_metrics: ModelMetrics
    output_scope: str = 'aggregate_only'
    governance_note: str = 'Individual employee risk ranking is disabled for consequential employment governance.'
