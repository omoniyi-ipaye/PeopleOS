from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional
from pydantic import BaseModel

from api.dependencies import AppState, get_app_state
from src.platform.provenance import IntegrityError, validated_risk_scores

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


def _model_lab(*, state: Optional[AppState] = None):
    """Resolve the optional predictive training stack only when requested.

    Everyday PeopleOS Desktop does not require XGBoost, LightGBM, Optuna,
    imbalanced-learn, SHAP, or Streamlit simply to start and run deterministic
    workforce analytics. Predictive tooling can be installed/enabled separately.
    """
    try:
        from src.model_lab_engine import ModelLabEngine
        if state is None:
            return ModelLabEngine()
        if not state.has_data() and not state.load_from_database():
            raise HTTPException(status_code=400, detail='No data loaded. Please upload a file first.')
        engine = getattr(state, 'ml_engine', None)
        if engine is not None and getattr(engine, 'is_trained', False):
            try:
                validated_risk_scores(state)
            except IntegrityError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
        return ModelLabEngine(ml_engine=engine, data=state.raw_df)
    except (ImportError, ModuleNotFoundError, OSError) as exc:
        raise HTTPException(
            status_code=409,
            detail=(
                "Predictive Model Lab is not installed in this PeopleOS runtime. "
                "Core workforce analytics remain available."
            ),
        ) from exc


@router.get("/validation", response_model=ValidationResponse)
async def get_model_validation(days_back: int = Query(default=90, ge=7, le=365)):
    """Run retroactive backtesting when the optional predictive stack exists."""
    result = _model_lab().backtest_flight_risk(days_back=days_back)
    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["message"])
    return result


@router.get("/sensitivity", response_model=List[FeatureSensitivity])
async def get_feature_sensitivity(state: AppState = Depends(get_app_state)):
    try:
        return _model_lab(state=state).analyze_feature_sensitivity()
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/refinement-plan", response_model=RefinementPlan)
async def get_refinement_plan(state: AppState = Depends(get_app_state)):
    try:
        return _model_lab(state=state).generate_refinement_plan()
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/optimize")
async def optimize_model(state: AppState = Depends(get_app_state)):
    try:
        plan = _model_lab(state=state).generate_refinement_plan()
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {
        "status": "review_only",
        "message": "Review generated; no model or feature changes were applied.",
        "plan_applied": False,
        "plan": plan,
    }
