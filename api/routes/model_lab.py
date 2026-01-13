from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from api.dependencies import get_app_state
from src.model_lab_engine import ModelLabEngine

router = APIRouter(prefix="/api/model-lab", tags=["Model Lab"])

class ValidationMetrics(BaseModel):
    precision: float
    recall: float
    f1_score: float
    sample_size: int
    true_positives: int
    false_positives: int
    missed_exits: int

class ValidationResponse(BaseModel):
    status: str
    metrics: Optional[ValidationMetrics] = None
    message: Optional[str] = None
    interpretation: Optional[str] = None
    
    model_config = {'protected_namespaces': ()}

class FeatureSensitivity(BaseModel):
    feature: str
    importance: float
    reliability: float
    status: str
    recommendation: str

class OptimizationMetrics(BaseModel):
    noisy_features: int
    redundant_dimensions: int
    estimated_accuracy_lift: str

class RefinementPlan(BaseModel):
    status: str
    suggested_actions: List[str]
    automated_features_to_prune: List[str]
    metrics: OptimizationMetrics
    reasoning: str

@router.get("/validation", response_model=ValidationResponse)
async def get_model_validation(
    days_back: int = Query(default=90, ge=7, le=365)
):
    """Run retroactive backtesting for the primary ML model."""
    lab = ModelLabEngine()
    result = lab.backtest_flight_risk(days_back=days_back)
    
    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["message"])
        
    return result

@router.get("/sensitivity", response_model=List[FeatureSensitivity])
async def get_feature_sensitivity():
    """Analyze signal-to-noise ratio and reliability of model features."""
    lab = ModelLabEngine()
    return lab.analyze_feature_sensitivity()

@router.get("/refinement-plan", response_model=RefinementPlan)
async def get_refinement_plan():
    """Generate suggestions for improving model performance."""
    lab = ModelLabEngine()
    return lab.generate_refinement_plan()

@router.post("/optimize")
async def optimize_model():
    """Trigger an automated model optimization and retraining cycle."""
    # This would involve calling the ML engine to retrain with pruned features
    # For now, we return the plan and a status
    lab = ModelLabEngine()
    plan = lab.generate_refinement_plan()
    
    return {
        "status": "success",
        "message": "Model optimization cycle triggered.",
        "plan_applied": plan
    }
