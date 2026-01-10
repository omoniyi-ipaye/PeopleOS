"""Predictions route handlers."""

import numpy as np
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel

from api.dependencies import get_app_state, AppState
from api.schemas.predictions import (
    ModelMetrics,
    FeatureImportance,
    FeatureImportanceResponse,
    RiskPrediction,
    RiskDistribution,
    EmployeeRiskDetail,
    PredictionsResponse,
)

router = APIRouter(prefix="/api/predictions", tags=["predictions"])


def require_predictions(state: AppState = Depends(get_app_state)) -> AppState:
    """Dependency that requires ML predictions to be available."""
    if not state.has_data():
        if not state.load_from_database():
            raise HTTPException(
                status_code=400,
                detail="No data loaded. Please upload a file first."
            )

    if state.ml_engine is None or not state.ml_engine.is_trained:
        raise HTTPException(
            status_code=400,
            detail="Predictive analytics not available. Upload data with Attrition column."
        )

    return state


@router.get("/model-metrics", response_model=ModelMetrics)
async def get_model_metrics(
    state: AppState = Depends(require_predictions)
) -> ModelMetrics:
    """
    Get ML model performance metrics.
    """
    if state.model_metrics is None:
        raise HTTPException(status_code=500, detail="Model metrics not available")

    return ModelMetrics(
        accuracy=state.model_metrics['accuracy'],
        precision=state.model_metrics['precision'],
        recall=state.model_metrics['recall'],
        f1=state.model_metrics['f1'],
        roc_auc=state.model_metrics.get('roc_auc'),
        best_model=state.model_metrics['best_model'],
        train_size=state.model_metrics['train_size'],
        test_size=state.model_metrics['test_size'],
        reliability=state.model_metrics.get('reliability', 'Unknown'),
        warnings=state.model_metrics.get('warnings')
    )


@router.get("/feature-importance", response_model=FeatureImportanceResponse)
async def get_feature_importance(
    limit: int = Query(default=10, ge=1, le=50, description="Number of features to return"),
    state: AppState = Depends(require_predictions)
) -> FeatureImportanceResponse:
    """
    Get feature importance rankings from the trained model.
    """
    importance_df = state.ml_engine.get_feature_importance_summary()

    if importance_df.empty:
        raise HTTPException(status_code=500, detail="Feature importance not available")

    features = []
    for _, row in importance_df.head(limit).iterrows():
        features.append(FeatureImportance(
            feature=row['Feature'],
            importance=float(row['Importance'])
        ))

    return FeatureImportanceResponse(
        features=features,
        model_name=state.ml_engine.best_model_name
    )


@router.get("/risk", response_model=PredictionsResponse)
async def get_all_risk_predictions(
    risk_category: Optional[str] = Query(default=None, description="Filter by risk category (High, Medium, Low)"),
    limit: int = Query(default=100, ge=1, le=1000, description="Maximum number of results"),
    offset: int = Query(default=0, ge=0, description="Offset for pagination"),
    state: AppState = Depends(require_predictions)
) -> PredictionsResponse:
    """
    Get risk predictions for all employees.
    """
    if state.risk_scores is None:
        raise HTTPException(status_code=500, detail="Risk scores not available")

    df = state.risk_scores.copy()

    # Filter by category if specified
    if risk_category:
        df = df[df['risk_category'] == risk_category]

    total = len(df)

    # Pagination
    df = df.iloc[offset:offset + limit]

    predictions = []
    for _, row in df.iterrows():
        predictions.append(RiskPrediction(
            employee_id=row['EmployeeID'],
            risk_score=float(row['risk_score']),
            risk_category=row['risk_category'],
            ci_lower=float(row['ci_lower']) if row['ci_lower'] else None,
            ci_upper=float(row['ci_upper']) if row['ci_upper'] else None
        ))

    # Calculate distribution
    all_df = state.risk_scores
    high_count = len(all_df[all_df['risk_category'] == 'High'])
    medium_count = len(all_df[all_df['risk_category'] == 'Medium'])
    low_count = len(all_df[all_df['risk_category'] == 'Low'])
    total_all = len(all_df)

    distribution = RiskDistribution(
        high_risk=high_count,
        medium_risk=medium_count,
        low_risk=low_count,
        total=total_all,
        high_risk_pct=round(high_count / total_all * 100, 1) if total_all > 0 else 0,
        medium_risk_pct=round(medium_count / total_all * 100, 1) if total_all > 0 else 0,
        low_risk_pct=round(low_count / total_all * 100, 1) if total_all > 0 else 0
    )

    return PredictionsResponse(
        predictions=predictions,
        distribution=distribution,
        model_metrics=ModelMetrics(
            accuracy=state.model_metrics['accuracy'],
            precision=state.model_metrics['precision'],
            recall=state.model_metrics['recall'],
            f1=state.model_metrics['f1'],
            roc_auc=state.model_metrics.get('roc_auc'),
            best_model=state.model_metrics['best_model'],
            train_size=state.model_metrics['train_size'],
            test_size=state.model_metrics['test_size'],
            reliability=state.model_metrics.get('reliability', 'Unknown'),
            warnings=state.model_metrics.get('warnings')
        )
    )


@router.get("/employee/{employee_id}", response_model=EmployeeRiskDetail)
async def get_employee_risk_detail(
    employee_id: str,
    state: AppState = Depends(require_predictions)
) -> EmployeeRiskDetail:
    """
    Get detailed risk analysis for a specific employee including SHAP drivers.
    """
    # Get employee data
    employee = state.get_employee_by_id(employee_id)
    if employee is None:
        raise HTTPException(status_code=404, detail=f"Employee {employee_id} not found")

    # Get risk data
    risk_data = state.get_employee_risk(employee_id)
    if risk_data is None:
        raise HTTPException(status_code=404, detail=f"Risk data not found for employee {employee_id}")

    # Get employee index for SHAP drivers
    emp_idx = state.get_employee_index(employee_id)
    if emp_idx is None:
        raise HTTPException(status_code=500, detail="Could not find employee index")

    # Get SHAP drivers
    drivers = state.ml_engine.get_risk_drivers(emp_idx, state.features_df)
    
    # Get recommendations
    recommendations = state.ml_engine.get_recommendations(
        employee_id,
        risk_data['risk_score'],
        drivers
    )

    # Get SHAP base value
    base_value = 0.5
    if state.ml_engine.shap_explainer is not None:
        try:
            bv = state.ml_engine.shap_explainer.expected_value
            if isinstance(bv, (list, np.ndarray)):
                base_value = float(bv[1]) if len(bv) > 1 else float(bv[0])
            else:
                base_value = float(bv)
        except Exception:
            pass

    return EmployeeRiskDetail(
        employee_id=employee_id,
        dept=employee['Dept'],
        tenure=float(employee['Tenure']),
        salary=float(employee['Salary']),
        last_rating=float(employee['LastRating']),
        age=int(employee['Age']),
        risk_score=risk_data['risk_score'],
        risk_category=risk_data['risk_category'],
        drivers=drivers[:10],  # Top 10 drivers
        recommendations=recommendations,
        base_value=base_value,
        confidence={
            'ci_lower': risk_data.get('ci_lower'),
            'ci_upper': risk_data.get('ci_upper'),
            'confidence_level': risk_data.get('confidence_level')
        }
    )


@router.get("/high-risk-employees")
async def get_high_risk_employees(
    limit: int = Query(default=20, ge=1, le=100, description="Maximum number of results"),
    state: AppState = Depends(require_predictions)
) -> dict:
    """
    Get top high-risk employees sorted by risk score.
    """
    if state.risk_scores is None or state.raw_df is None:
        raise HTTPException(status_code=500, detail="Risk scores not available")

    # Get high risk employees
    high_risk = state.risk_scores[state.risk_scores['risk_category'] == 'High'].copy()
    high_risk = high_risk.sort_values('risk_score', ascending=False).head(limit)

    # Merge with employee data
    merged = high_risk.merge(
        state.raw_df[['EmployeeID', 'Dept', 'Tenure', 'Salary', 'LastRating']],
        on='EmployeeID',
        how='left'
    )

    employees = []
    for _, row in merged.iterrows():
        employees.append({
            'employee_id': row['EmployeeID'],
            'dept': row['Dept'],
            'tenure': float(row['Tenure']),
            'salary': float(row['Salary']),
            'last_rating': float(row['LastRating']),
            'risk_score': float(row['risk_score']),
            'risk_category': row['risk_category']
        })

    return {
        'employees': employees,
        'total_high_risk': len(state.risk_scores[state.risk_scores['risk_category'] == 'High'])
    }
