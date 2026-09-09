"""Governed descriptive analytics routes.

Current-state metrics preserve population/denominator semantics. Correlations and
group tests are observational. Time-series forecasts require real dated repeated
snapshots; PeopleOS never fabricates historical observations from current fields.
Individual cluster membership is not exposed.
"""

from typing import Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query

from api.dependencies import AppState, get_app_state
from api.schemas.analytics import (
    AgeDistribution,
    AnalyticsSummary,
    CorrelationData,
    CorrelationsResponse,
    DepartmentList,
    DepartmentStats,
    DistributionsResponse,
    HighRiskDepartment,
    HighRiskDepartmentsResponse,
    SalaryBand,
    TenureDistribution,
)

from src.serialization import json_safe

router = APIRouter(prefix="/api/analytics", tags=["analytics"])


def require_data(state: AppState = Depends(get_app_state)) -> AppState:
    if not state.has_data():
        if not state.load_from_database():
            raise HTTPException(status_code=400, detail="No data loaded. Please upload a file first.")
    return state


@router.get("/summary", response_model=AnalyticsSummary)
async def get_analytics_summary(state: AppState = Depends(require_data)) -> AnalyticsSummary:
    if state.analytics_engine is None:
        raise HTTPException(status_code=500, detail="Analytics engine not initialized")
    stats = state.analytics_engine.get_summary_statistics()
    share = stats.get('observed_attrition_share', stats.get('turnover_rate'))
    takeaways = []
    if share is not None:
        takeaways.append(f"Observed attrition share in the current outcome population is {share:.1%}; this is not a period turnover rate.")
    return AnalyticsSummary(
        headcount=stats.get('headcount', 0),
        record_count=stats.get('record_count'),
        observed_attrition_share=share,
        turnover_rate=share,
        turnover_rate_semantics='observed_attrition_share_not_period_turnover',
        department_count=stats.get('department_count', 0),
        salary_mean=stats.get('salary_mean'),
        salary_median=stats.get('salary_median'),
        tenure_mean=stats.get('tenure_mean'),
        tenure_median=stats.get('tenure_median'),
        age_mean=stats.get('age_mean'),
        lastrating_mean=stats.get('lastrating_mean'),
        attrition_count=stats.get('attrition_count'),
        active_count=stats.get('active_count'),
        attrition_known_count=stats.get('attrition_known_count'),
        salary_observations=stats.get('salary_observations'),
        salary_excluded_count=stats.get('salary_excluded_count'),
        tenure_observations=stats.get('tenure_observations'),
        tenure_excluded_count=stats.get('tenure_excluded_count'),
        age_observations=stats.get('age_observations'),
        age_excluded_count=stats.get('age_excluded_count'),
        lastrating_observations=stats.get('lastrating_observations'),
        lastrating_excluded_count=stats.get('lastrating_excluded_count'),
        takeaways=takeaways,
        insights={'attrition_semantics': 'Observed snapshot outcome share; period turnover requires a defined period and at-risk denominator.'},
    )


@router.get("/departments", response_model=DepartmentList)
async def get_department_stats(state: AppState = Depends(require_data)) -> DepartmentList:
    frame = state.analytics_engine.get_department_aggregates()
    minimum_group_size = 10
    eligible = frame['Total_Records'].fillna(0) >= minimum_group_size if 'Total_Records' in frame else pd.Series(False, index=frame.index)
    suppressed = int((~eligible).sum())
    frame = frame.loc[eligible]
    departments = []
    for _, row in frame.iterrows():
        row = json_safe(row.to_dict())
        share = row.get('Observed_Attrition_Share', row.get('Turnover_Rate'))
        departments.append(DepartmentStats(
            dept=row['Dept'],
            headcount=int(row.get('Headcount', 0)),
            total_records=int(row.get('Total_Records', 0)),
            outcome_observations=int(row.get('Outcome_Observations', 0)),
            avg_salary=row.get('Avg_Salary'),
            median_salary=row.get('Median_Salary'),
            salary_std_dev=row.get('Salary_StdDev'),
            avg_tenure=row.get('Avg_Tenure'),
            avg_rating=row.get('Avg_Rating'),
            avg_age=row.get('Avg_Age'),
            observed_attrition_share=share,
            turnover_rate=share,
        ))
    return DepartmentList(departments=departments, total_departments=len(departments),
                          minimum_group_size=minimum_group_size,
                          suppressed_department_count=suppressed)


@router.get("/distributions", response_model=DistributionsResponse)
async def get_distributions(state: AppState = Depends(require_data)) -> DistributionsResponse:
    tenure_frame = state.analytics_engine.get_tenure_distribution()
    tenure = [
        TenureDistribution(
            tenure_range=str(row['Tenure_Range']),
            count=int(row['Count']),
            observed_attrition_share=json_safe(row.get('Observed_Attrition_Share', row.get('Turnover_Rate'))),
            turnover_rate=json_safe(row.get('Observed_Attrition_Share', row.get('Turnover_Rate'))),
        ) for _, row in tenure_frame.iterrows()
    ]
    age_frame = state.analytics_engine.get_age_distribution()
    age = [AgeDistribution(age_range=str(row['Age_Range']), count=int(row['Count'])) for _, row in age_frame.iterrows()]
    salary_frame = state.analytics_engine.get_salary_bands()
    salary = [SalaryBand(band=row['Band'], lower=float(row['Lower']), upper=float(row['Upper']), count=int(row['Count'])) for _, row in salary_frame.iterrows()]
    return DistributionsResponse(tenure=tenure, age=age, salary_bands=salary)


@router.get("/correlations", response_model=CorrelationsResponse)
async def get_correlations(
    target: str = Query(default="Attrition", description="Target for observational correlation screening"),
    limit: int = Query(default=10, ge=1, le=50),
    state: AppState = Depends(require_data),
) -> CorrelationsResponse:
    frame = state.analytics_engine.get_correlations(target_column=target)
    if frame.empty:
        return CorrelationsResponse(correlations=[], target_column=target)
    frame = frame.dropna(subset=['Correlation', 'Abs_Correlation'])
    items = [CorrelationData(feature=row['Feature'], correlation=float(row['Correlation']), abs_correlation=float(row['Abs_Correlation'])) for _, row in frame.head(limit).iterrows()]
    return CorrelationsResponse(correlations=items, target_column=target)


@router.get("/high-risk-departments", response_model=HighRiskDepartmentsResponse)
async def get_high_risk_departments(
    threshold: Optional[float] = Query(default=None, ge=0, le=1, description="Observed attrition-share screening threshold"),
    state: AppState = Depends(require_data),
) -> HighRiskDepartmentsResponse:
    minimum_group_size = 10
    used = state.analytics_engine.high_risk_threshold if threshold is None else threshold

    # Inspect the full department denominator first. An empty hotspot frame means
    # something very different when no outcomes are measured at all.
    all_departments = state.analytics_engine.get_department_aggregates()
    if all_departments is None or all_departments.empty or 'Outcome_Observations' not in all_departments.columns:
        return HighRiskDepartmentsResponse(
            departments=[], threshold=used, minimum_group_size=minimum_group_size,
            suppressed_department_count=0, outcome_observations=0, evidence_available=False,
            unavailable_reason='Department outcome observations are unavailable; an empty hotspot list is not evidence that no hotspot exists.',
        )
    outcome_observations = int(pd.to_numeric(all_departments['Outcome_Observations'], errors='coerce').fillna(0).sum())
    if outcome_observations == 0:
        return HighRiskDepartmentsResponse(
            departments=[], threshold=used, minimum_group_size=minimum_group_size,
            suppressed_department_count=0, outcome_observations=0, evidence_available=False,
            unavailable_reason='No known department attrition outcomes are present; PeopleOS will not interpret missing outcome evidence as no hotspots.',
        )

    frame = state.analytics_engine.get_high_risk_departments(threshold=threshold)
    eligible = frame['Total_Records'].fillna(0) >= minimum_group_size if 'Total_Records' in frame else pd.Series(False, index=frame.index)
    suppressed = int((~eligible).sum())
    frame = frame.loc[eligible]
    departments = []
    for _, row in frame.iterrows():
        share = float(row.get('Observed_Attrition_Share', row.get('Turnover_Rate', 0)))
        departments.append(HighRiskDepartment(
            dept=row['Dept'], observed_attrition_share=share, turnover_rate=share,
            headcount=int(row.get('Headcount', 0)), avg_salary=row.get('Avg_Salary'), avg_rating=row.get('Avg_Rating'),
            reason='Observed attrition share exceeds the configured aggregate screening threshold; local causes are not inferred.',
        ))
    return HighRiskDepartmentsResponse(
        departments=departments, threshold=used,
        minimum_group_size=minimum_group_size,
        suppressed_department_count=suppressed,
        outcome_observations=outcome_observations,
        evidence_available=True,
        unavailable_reason=None,
    )


@router.get("/clusters")
async def get_clusters(state: AppState = Depends(require_data)):
    """Disabled until cluster semantics and aggregate-only outputs are explicitly governed."""
    raise HTTPException(status_code=409, detail="Employee clustering is disabled in the governed product boundary pending validated use-case and aggregate-only semantics.")


@router.get("/cluster-members/{cluster_id}", deprecated=True)
async def get_cluster_members(cluster_id: int, state: AppState = Depends(require_data)):
    raise HTTPException(status_code=403, detail="Employee cluster membership is not exposed by PeopleOS.")


@router.get("/forecast")
async def get_forecast(metric: str = "headcount", periods: int = Query(default=12, ge=1, le=36), state: AppState = Depends(require_data)):
    """Forecast only from genuine repeated dated snapshots; never synthetic backfill."""
    history = getattr(state, 'historical_df', None)
    if history is None or history.empty:
        return {'success': False, 'reason': 'Forecasting requires observed monthly workforce censuses'}
    from src.forecasting_engine import ForecastingEngine
    return ForecastingEngine(history).forecast_metric(metric, periods)


@router.get("/compare-groups")
async def compare_groups(
    group_by: str = Query(...), metric: str = Query(...), state: AppState = Depends(require_data),
):
    result = state.analytics_engine.compare_groups(group_by, metric)
    if not result.get('success'):
        raise HTTPException(status_code=400, detail=result.get('reason', 'Comparison failed'))
    significant = bool(result.get('is_significant'))
    result['interpretation_boundary'] = (
        f"A statistically {'detectable' if significant else 'non-detectable'} mean difference was observed for {metric} across {group_by} in this sample. "
        "Statistical significance alone does not establish unfairness, causation, practical importance, or that no issue exists."
    )
    result['recommended_action'] = 'Review effect size, group sizes, role/level mix, confounding and business context before drawing conclusions.'
    return result
