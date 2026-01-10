"""
API routes for Survival Analysis endpoints.
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Optional, List

from api.dependencies import get_app_state, AppState
from api.schemas.survival import (
    SurvivalAnalysisResponse,
    CoxModelResult,
    CohortInsight,
    AtRiskEmployee,
    EmployeeSurvivalPrediction,
    HazardOverTime,
)

router = APIRouter(prefix="/api/survival", tags=["survival"])


def require_survival(state: AppState = Depends(get_app_state)) -> AppState:
    """Dependency that requires survival engine to be available."""
    if not state.has_data():
        if not state.load_from_database():
            raise HTTPException(
                status_code=400,
                detail="No data loaded. Please upload a file first."
            )

    if state.survival_engine is None:
        raise HTTPException(
            status_code=400,
            detail="Survival analysis not available. Ensure data has Tenure and Attrition columns."
        )

    return state


@router.get("/analysis", response_model=SurvivalAnalysisResponse)
async def get_survival_analysis(
    state: AppState = Depends(require_survival)
) -> SurvivalAnalysisResponse:
    """
    Get complete survival analysis.

    Includes:
    - Kaplan-Meier survival curves
    - Cox Proportional Hazards model
    - Cohort insights
    - At-risk employees
    - Recommendations
    """
    results = state.survival_engine.analyze_all()

    return SurvivalAnalysisResponse(
        kaplan_meier=results.get('kaplan_meier'),
        kaplan_meier_by_dept=results.get('kaplan_meier_by_dept'),
        cox_model=CoxModelResult(**results.get('cox_model', {})) if results.get('cox_model') else None,
        hazard_over_time=HazardOverTime(**results.get('hazard_over_time', {})) if results.get('hazard_over_time') else None,
        cohort_insights=[CohortInsight(**c) for c in results.get('cohort_insights', [])],
        at_risk_employees=[AtRiskEmployee(**e) for e in results.get('at_risk_employees', [])],
        summary=results.get('summary', {}),
        recommendations=results.get('recommendations', []),
        warnings=results.get('warnings', [])
    )


@router.get("/kaplan-meier")
async def get_kaplan_meier(
    segment_by: Optional[str] = Query(
        default=None,
        description="Column to segment by (e.g., 'Dept', 'Location')"
    ),
    state: AppState = Depends(require_survival)
):
    """
    Get Kaplan-Meier survival curves.

    Optionally segment by department, location, or other categorical columns.
    """
    results = state.survival_engine.fit_kaplan_meier(segment_by=segment_by)
    return results


@router.get("/cox-model", response_model=CoxModelResult)
async def get_cox_model(
    state: AppState = Depends(require_survival)
) -> CoxModelResult:
    """
    Get Cox Proportional Hazards model results.

    Returns hazard ratios and statistical significance for each covariate.
    """
    results = state.survival_engine.fit_cox_proportional_hazards()
    return CoxModelResult(**results)


@router.get("/hazard-over-time", response_model=HazardOverTime)
async def get_hazard_over_time(
    state: AppState = Depends(require_survival)
) -> HazardOverTime:
    """
    Get baseline hazard function over time.

    Shows when attrition risk is highest during the employee lifecycle.
    """
    results = state.survival_engine.get_hazard_over_time()
    return HazardOverTime(**results)


@router.get("/cohort-insights", response_model=CohortInsight)
async def get_cohort_insights(
    dept: Optional[str] = Query(default=None, description="Filter by department"),
    location: Optional[str] = Query(default=None, description="Filter by location"),
    tenure_min: Optional[float] = Query(default=None, description="Minimum tenure in years"),
    tenure_max: Optional[float] = Query(default=None, description="Maximum tenure in years"),
    years_since_promotion_min: Optional[float] = Query(
        default=None,
        description="Minimum years since last promotion"
    ),
    state: AppState = Depends(require_survival)
) -> CohortInsight:
    """
    Get survival insights for a filtered employee cohort.

    Example: Engineers in UK with >2 years tenure and no promotion in 18+ months.
    """
    filters = {}
    if dept:
        filters['Dept'] = dept
    if location:
        filters['Location'] = location
    if tenure_min:
        filters['tenure_min'] = tenure_min
    if tenure_max:
        filters['tenure_max'] = tenure_max
    if years_since_promotion_min:
        filters['years_since_promotion_min'] = years_since_promotion_min

    results = state.survival_engine.generate_cohort_insights(filters=filters if filters else None)
    return CohortInsight(**results)


@router.get("/at-risk", response_model=List[AtRiskEmployee])
async def get_at_risk_employees(
    limit: int = Query(default=20, ge=1, le=100, description="Number of employees to return"),
    state: AppState = Depends(require_survival)
) -> List[AtRiskEmployee]:
    """
    Get employees with highest attrition risk.

    Returns employees ranked by 12-month attrition probability.
    """
    at_risk_df = state.survival_engine.get_at_risk_employees(top_n=limit)

    if at_risk_df.empty:
        return []

    employees = []
    for _, row in at_risk_df.iterrows():
        employees.append(AtRiskEmployee(
            EmployeeID=row['EmployeeID'],
            survival_3mo=row.get('survival_3mo'),
            survival_6mo=row.get('survival_6mo'),
            survival_12mo=row.get('survival_12mo'),
            attrition_risk_3mo=row.get('attrition_risk_3mo'),
            attrition_risk_6mo=row.get('attrition_risk_6mo'),
            attrition_risk_12mo=row.get('attrition_risk_12mo'),
            risk_category=row.get('risk_category', 'Unknown'),
            current_tenure_years=row.get('current_tenure_years'),
            current_rating=row.get('current_rating'),
            Dept=row.get('Dept'),
            Location=row.get('Location'),
            JobTitle=row.get('JobTitle'),
            YearsSinceLastPromotion=row.get('YearsSinceLastPromotion'),
            CompaRatio=row.get('CompaRatio'),
            risk_factors=row.get('risk_factors')
        ))

    return employees


@router.get("/employee/{employee_id}", response_model=EmployeeSurvivalPrediction)
async def get_employee_survival(
    employee_id: str,
    state: AppState = Depends(require_survival)
) -> EmployeeSurvivalPrediction:
    """
    Get survival prediction for a specific employee.

    Returns probability of staying at 3, 6, 12, and 24 month horizons.
    """
    predictions_df = state.survival_engine.predict_survival_probability(
        employee_ids=[employee_id]
    )

    if predictions_df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No survival prediction available for employee {employee_id}"
        )

    row = predictions_df.iloc[0]
    return EmployeeSurvivalPrediction(
        EmployeeID=row['EmployeeID'],
        survival_3mo=row.get('survival_3mo', 0),
        survival_6mo=row.get('survival_6mo', 0),
        survival_12mo=row.get('survival_12mo', 0),
        survival_24mo=row.get('survival_24mo'),
        attrition_risk_12mo=row.get('attrition_risk_12mo', 0),
        risk_category=row.get('risk_category', 'Unknown'),
        current_tenure_years=row.get('current_tenure_years', 0),
        risk_factors=row.get('risk_factors')
    )
