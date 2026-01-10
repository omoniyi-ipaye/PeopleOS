"""Analytics route handlers."""

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel

from api.dependencies import get_app_state, AppState
from api.schemas.analytics import (
    AnalyticsSummary,
    DepartmentStats,
    DepartmentList,
    TenureDistribution,
    AgeDistribution,
    SalaryBand,
    CorrelationData,
    HighRiskDepartment,
    DistributionsResponse,
    CorrelationsResponse,
    HighRiskDepartmentsResponse,
)

router = APIRouter(prefix="/api/analytics", tags=["analytics"])


def require_data(state: AppState = Depends(get_app_state)) -> AppState:
    """Dependency that requires data to be loaded."""
    if not state.has_data():
        # Try loading from database first
        from src.logger import get_logger
        logger = get_logger('analytics_route')
        logger.info("No in-memory data found, attempting database load")
        
        if not state.load_from_database():
            logger.warning("Database load failed - returning 400 (No data loaded)")
            raise HTTPException(
                status_code=400,
                detail="No data loaded. Please upload a file first."
            )
        logger.info("Successfully recovered data from database")
    return state


@router.get("/summary", response_model=AnalyticsSummary)
async def get_analytics_summary(
    state: AppState = Depends(require_data)
) -> AnalyticsSummary:
    """
    Get overall analytics summary including headcount, turnover, and key metrics.
    """
    if state.analytics_engine is None:
        raise HTTPException(status_code=500, detail="Analytics engine not initialized")

    stats = state.analytics_engine.get_summary_statistics()

    return AnalyticsSummary(
        headcount=stats.get('headcount', 0),
        turnover_rate=stats.get('turnover_rate'),
        department_count=stats.get('department_count', 0),
        salary_mean=stats.get('salary_mean'),
        salary_median=stats.get('salary_median'),
        tenure_mean=stats.get('tenure_mean'),
        tenure_median=stats.get('tenure_median'),
        age_mean=stats.get('age_mean'),
        lastrating_mean=stats.get('lastrating_mean'),
        attrition_count=stats.get('attrition_count'),
        active_count=stats.get('active_count'),
        takeaways=state.insight_interpreter.get_key_takeaways(stats) if state.insight_interpreter else [],
        insights={
            'headcount': state.insight_interpreter.interpret_metric('headcount', stats.get('headcount', 0)),
            'turnover_rate': state.insight_interpreter.interpret_metric('turnover_rate', stats.get('turnover_rate')),
            'tenure_mean': state.insight_interpreter.interpret_metric('tenure_mean', stats.get('tenure_mean')),
            'lastrating_mean': state.insight_interpreter.interpret_metric('lastrating_mean', stats.get('lastrating_mean')),
        } if state.insight_interpreter else {}
    )


@router.get("/departments", response_model=DepartmentList)
async def get_department_stats(
    state: AppState = Depends(require_data)
) -> DepartmentList:
    """
    Get department-level analytics including salary, tenure, and turnover.
    """
    if state.analytics_engine is None:
        raise HTTPException(status_code=500, detail="Analytics engine not initialized")

    dept_df = state.analytics_engine.get_department_aggregates()

    departments = []
    for _, row in dept_df.iterrows():
        departments.append(DepartmentStats(
            dept=row['Dept'],
            headcount=int(row.get('Headcount', 0)),
            avg_salary=row.get('Avg_Salary'),
            median_salary=row.get('Median_Salary'),
            salary_std_dev=row.get('Salary_StdDev'),
            avg_tenure=row.get('Avg_Tenure'),
            avg_rating=row.get('Avg_Rating'),
            avg_age=row.get('Avg_Age'),
            turnover_rate=row.get('Turnover_Rate')
        ))

    return DepartmentList(
        departments=departments,
        total_departments=len(departments)
    )


@router.get("/distributions", response_model=DistributionsResponse)
async def get_distributions(
    state: AppState = Depends(require_data)
) -> DistributionsResponse:
    """
    Get tenure, age, and salary distributions.
    """
    if state.analytics_engine is None:
        raise HTTPException(status_code=500, detail="Analytics engine not initialized")

    # Tenure distribution
    tenure_df = state.analytics_engine.get_tenure_distribution()
    tenure_dist = []
    for _, row in tenure_df.iterrows():
        tenure_dist.append(TenureDistribution(
            tenure_range=str(row['Tenure_Range']),
            count=int(row['Count']),
            turnover_rate=row.get('Turnover_Rate')
        ))

    # Age distribution
    age_df = state.analytics_engine.get_age_distribution()
    age_dist = []
    for _, row in age_df.iterrows():
        age_dist.append(AgeDistribution(
            age_range=str(row['Age_Range']),
            count=int(row['Count'])
        ))

    # Salary bands
    salary_df = state.analytics_engine.get_salary_bands()
    salary_bands = []
    for _, row in salary_df.iterrows():
        salary_bands.append(SalaryBand(
            band=row['Band'],
            lower=float(row['Lower']),
            upper=float(row['Upper']),
            count=int(row['Count'])
        ))

    return DistributionsResponse(
        tenure=tenure_dist,
        age=age_dist,
        salary_bands=salary_bands
    )


@router.get("/correlations", response_model=CorrelationsResponse)
async def get_correlations(
    target: str = Query(default="Attrition", description="Target column for correlation"),
    limit: int = Query(default=10, ge=1, le=50, description="Number of features to return"),
    state: AppState = Depends(require_data)
) -> CorrelationsResponse:
    """
    Get feature correlations with the target column (default: Attrition).
    """
    if state.analytics_engine is None:
        raise HTTPException(status_code=500, detail="Analytics engine not initialized")

    corr_df = state.analytics_engine.get_correlations(target_column=target)

    if corr_df.empty:
        return CorrelationsResponse(
            correlations=[],
            target_column=target
        )

    correlations = []
    for _, row in corr_df.head(limit).iterrows():
        correlations.append(CorrelationData(
            feature=row['Feature'],
            correlation=float(row['Correlation']),
            abs_correlation=float(row['Abs_Correlation'])
        ))

    return CorrelationsResponse(
        correlations=correlations,
        target_column=target
    )


@router.get("/high-risk-departments", response_model=HighRiskDepartmentsResponse)
async def get_high_risk_departments(
    threshold: Optional[float] = Query(default=None, ge=0, le=1, description="Turnover rate threshold"),
    state: AppState = Depends(require_data)
) -> HighRiskDepartmentsResponse:
    """
    Get departments with turnover rate above threshold.
    """
    if state.analytics_engine is None:
        raise HTTPException(status_code=500, detail="Analytics engine not initialized")

    high_risk_df = state.analytics_engine.get_high_risk_departments(threshold=threshold)
    used_threshold = threshold or state.analytics_engine.high_risk_threshold

    departments = []
    for _, row in high_risk_df.iterrows():
        reasons = []
        if float(row['Turnover_Rate']) > used_threshold * 2.0:
            reasons.append("Critical attrition levels detected")
        elif float(row['Turnover_Rate']) > used_threshold * 1.5:
            reasons.append("Significantly elevated turnover")
        
        salary_mean = state.analytics_engine.get_summary_statistics().get('salary_mean')
        if row.get('Avg_Salary') and salary_mean:
            if float(row['Avg_Salary']) < salary_mean * 0.75:
                reasons.append("Severe compensation gap vs organization average")
            elif float(row['Avg_Salary']) < salary_mean * 0.85:
                reasons.append("Below-market compensation band")
        
        if row.get('Avg_Rating') and float(row['Avg_Rating']) < 2.8:
            reasons.append("Systemic low performance scores")
        elif row.get('Avg_Rating') and float(row['Avg_Rating']) < 3.2:
            reasons.append("Sub-par team performance metrics")
        
        if row.get('Avg_Tenure') and float(row['Avg_Tenure']) < 1.5:
            reasons.append("Recent hiring surge or low long-term retention")

        reason = "; ".join(reasons) if reasons else "Unusual turnover volatility detected"

        departments.append(HighRiskDepartment(
            dept=row['Dept'],
            turnover_rate=float(row['Turnover_Rate']),
            headcount=int(row.get('Headcount', 0)),
            avg_salary=row.get('Avg_Salary'),
            avg_rating=row.get('Avg_Rating'),
            reason=reason
        ))

    return HighRiskDepartmentsResponse(
        departments=departments,
        threshold=used_threshold
    )
