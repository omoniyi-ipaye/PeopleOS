"""Governed predictive analytics routes.

Predictive outputs are aggregate-first. Individual employee risk rankings and
individual recommendation endpoints are deliberately disabled because they can
be misused for consequential employment decisions.
"""

from src.platform.provenance import IntegrityError, validated_risk_scores

from fastapi import APIRouter, Depends, HTTPException, Query

from api.dependencies import AppState, get_app_state
from api.schemas.predictions import FeatureImportance, FeatureImportanceResponse, ModelMetrics, PredictionsResponse, RiskDistribution

router = APIRouter(prefix='/api/predictions', tags=['predictions'])


def require_predictions(state: AppState = Depends(get_app_state)) -> AppState:
    if not state.has_data():
        if not state.load_from_database():
            raise HTTPException(status_code=400, detail='No data loaded. Please upload a file first.')
    try:
        validated_risk_scores(state)
    except IntegrityError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return state


def _model_metrics(state: AppState) -> ModelMetrics:
    metrics = state.model_metrics or {}
    return ModelMetrics(
        accuracy=metrics['accuracy'], precision=metrics['precision'], recall=metrics['recall'], f1=metrics['f1'],
        roc_auc=metrics.get('roc_auc'), brier_score=metrics.get('brier_score'), calibration_error=metrics.get('calibration_error'),
        best_model=metrics['best_model'], train_size=metrics['train_size'], test_size=metrics['test_size'],
        reliability=metrics.get('reliability', 'Unknown'), warnings=metrics.get('warnings'),
        baseline_brier_score=metrics.get('baseline_brier_score'), brier_skill_score=metrics.get('brier_skill_score'),
        average_precision=metrics.get('average_precision'), baseline_average_precision=metrics.get('baseline_average_precision'),
        validation_checks=metrics.get('validation_checks'),
        evaluation_semantics=metrics.get('evaluation_semantics', 'unvalidated'),
        future_departure_validated=metrics.get('future_departure_validated', False),
    )


@router.get('/model-metrics', response_model=ModelMetrics)
async def get_model_metrics(state: AppState = Depends(require_predictions)) -> ModelMetrics:
    return _model_metrics(state)


@router.get('/feature-importance', response_model=FeatureImportanceResponse)
async def get_feature_importance(
    limit: int = Query(default=10, ge=1, le=50),
    state: AppState = Depends(require_predictions),
) -> FeatureImportanceResponse:
    importance_df = state.ml_engine.get_feature_importance_summary()
    if importance_df.empty:
        raise HTTPException(status_code=404, detail='Feature importance is not available for the activated model.')
    features = [FeatureImportance(feature=row['feature'], importance=float(row['importance'])) for _, row in importance_df.head(limit).iterrows()]
    return FeatureImportanceResponse(features=features, model_name=state.ml_engine.best_model_name)


@router.get('/risk', response_model=PredictionsResponse)
async def get_aggregate_risk_distribution(
    risk_category: str | None = Query(default=None, description='Deprecated; aggregate distribution is always returned.'),
    limit: int = Query(default=100, ge=1, le=1000, description='Deprecated compatibility parameter.'),
    offset: int = Query(default=0, ge=0, description='Deprecated compatibility parameter.'),
    state: AppState = Depends(require_predictions),
) -> PredictionsResponse:
    try:
        frame = validated_risk_scores(state)
    except IntegrityError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    high = int((frame['risk_category'] == 'High').sum())
    medium = int((frame['risk_category'] == 'Medium').sum())
    low = int((frame['risk_category'] == 'Low').sum())
    total = len(frame)
    metrics = _model_metrics(state)
    distribution = RiskDistribution(
        high_risk=high, medium_risk=medium, low_risk=low, total=total,
        high_risk_pct=round(high / total * 100, 1) if total else 0,
        medium_risk_pct=round(medium / total * 100, 1) if total else 0,
        low_risk_pct=round(low / total * 100, 1) if total else 0,
        category_semantics=(
            'configured_model_score_bands_with_prospective_future_departure_validation'
            if metrics.future_departure_validated else
            'configured_model_score_bands_not_validated_future_departure_probabilities'
        ),
        future_departure_validated=metrics.future_departure_validated,
    )
    # Individual rows are intentionally not returned even though the compatibility
    # schema retains the field.
    return PredictionsResponse(predictions=[], distribution=distribution, model_metrics=metrics)


@router.get('/employee/{employee_id}')
async def get_employee_risk_detail(employee_id: str, state: AppState = Depends(require_predictions)):
    raise HTTPException(
        status_code=403,
        detail='Individual predictive risk views are disabled. Use aggregate Retention Signals and governed investigation instead.',
    )


@router.get('/high-risk-employees')
async def get_high_risk_employees(
    limit: int = Query(default=20, ge=1, le=100),
    state: AppState = Depends(require_predictions),
):
    raise HTTPException(
        status_code=403,
        detail='Employee risk ranking is disabled. Predictive analytics may prioritize aggregate investigation, not consequential individual action.',
    )
