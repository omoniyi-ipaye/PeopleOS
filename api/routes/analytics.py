"""Analytics route handlers."""

from typing import List, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel

from src.database import Database
from src.clustering_engine import ClusteringEngine
from src.forecasting_engine import ForecastingEngine
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

from src.logger import get_logger

router = APIRouter(prefix="/api/analytics", tags=["analytics"])
logger = get_logger("analytics_router")


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

    # FIX: Drop NaN values before serialization to prevent JSON error
    corr_df = corr_df.dropna(subset=['Correlation', 'Abs_Correlation'])

    correlations = []
    for _, row in corr_df.head(limit).iterrows():
        # Additional NaN check for safety
        corr_val = row['Correlation']
        abs_corr_val = row['Abs_Correlation']
        if pd.isna(corr_val) or pd.isna(abs_corr_val):
            continue
        correlations.append(CorrelationData(
            feature=row['Feature'],
            correlation=float(corr_val),
            abs_correlation=float(abs_corr_val)
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


@router.get("/clusters")
async def get_clusters(n_clusters: int = 4, auto_tune: bool = False, state: AppState = Depends(require_data)):
    """
    Get employee segmentation clusters using unsupervised learning.

    Args:
        n_clusters: Number of clusters to create (default: 4)
        auto_tune: If True, automatically finds optimal n_clusters (default: False for consistency)
    """
    try:
        # Use app state's DataFrame if available, otherwise get from database
        if state.analytics_engine is not None:
            df = state.analytics_engine.df.copy()
        else:
            db = Database()
            df = db.get_all_employees()

        if df.empty:
            raise HTTPException(status_code=400, detail="No employee data available for clustering")

        cluster_engine = ClusteringEngine(df)
        result = cluster_engine.train(n_clusters=n_clusters, auto_tune=auto_tune)

        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cluster-members/{cluster_id}")
async def get_cluster_members(cluster_id: int, n_clusters: int = 4, state: AppState = Depends(require_data)):
    """
    Get the list of employees belonging to a specific cluster.

    Args:
        cluster_id: The cluster ID to get members for
        n_clusters: Number of clusters (must match the n_clusters used in /clusters endpoint)
    """
    try:
        if state.analytics_engine is not None:
            df = state.analytics_engine.df.copy()
        else:
            db = Database()
            df = db.get_all_employees()

        cluster_engine = ClusteringEngine(df)
        # Use auto_tune=False to ensure consistent clustering with the same n_clusters
        result = cluster_engine.train(n_clusters=n_clusters, auto_tune=False)

        if not result['success']:
            raise HTTPException(status_code=400, detail="Clustering failed: " + result.get('reason', 'Unknown error'))

        labels = result['labels']
        member_ids = [emp_id for emp_id, c_id in labels.items() if int(c_id) == int(cluster_id)]

        # Filter and only return basic info - handle missing columns gracefully
        available_cols = ['EmployeeID']
        for col in ['Dept', 'JobTitle', 'Salary', 'LastRating']:
            if col in df.columns:
                available_cols.append(col)

        members = df[df['EmployeeID'].isin(member_ids)][available_cols].to_dict('records')

        return {
            "success": True,
            "cluster_id": cluster_id,
            "count": len(members),
            "members": members
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/forecast")
async def get_forecast(metric: str = "headcount", periods: int = 12, state: AppState = Depends(require_data)):
    """
    Get time-series forecast for key metrics.
    """
    try:
        db = Database()
        
        # Get historical snapshots
        history_df = db.get_historical_snapshots(start_date="2000-01-01")
        
        # Determine if we should use synthetic history fallback
        use_synthetic = history_df.empty
        if not use_synthetic:
            # Check if history has sufficient temporal spread (at least 30 days)
            history_df['snapshot_date'] = pd.to_datetime(history_df['snapshot_date'])
            date_range = (history_df['snapshot_date'].max() - history_df['snapshot_date'].min()).days
            if date_range < 30:
                logger.info(f"Snapshots only cover {date_range} days. Using synthetic fallback for better trends.")
                use_synthetic = True
        
        if metric == "headcount" and use_synthetic:
            # Use HireDate-based headcount if no/few snapshots exist
            logger.info("Using HireDate for synthetic headcount trend.")
            ts_df = db.get_synthetic_historical_headcount(start_date="2020-01-01")
            
            if not ts_df.empty:
                ts = ts_df.set_index('date')['headcount']
                # Add current headcount as the final point to ensure accuracy
                current_count = db.get_employee_count()
                ts[pd.Timestamp.now().normalize()] = current_count
                ts = ts.sort_index()
            else:
                return {"success": False, "reason": "No data found to generate headcount trend."}
        elif use_synthetic:
            if metric == "salary":
                logger.info("Using Tenure-based synthetic salary trend.")
                ts_df = db.get_synthetic_historical_salary(start_date="2020-01-01")
                if ts_df.empty:
                    return {"success": False, "reason": "No data found to generate salary trend."}
                ts = ts_df.set_index('date')['value']
            else:
                return {
                    "success": False, 
                    "reason": f"No historical snapshots found for {metric}. Upload data with 'SnapshotDate' to backfill history."
                }
        else:
            # Standardize dates
            history_df['snapshot_date'] = pd.to_datetime(history_df['snapshot_date'])
            
            metric = metric.lower()
            # Aggregate by date
            if metric == "headcount":
                ts = history_df.groupby('snapshot_date')['EmployeeID'].nunique()
            elif metric == "salary":
                ts = history_df.groupby('snapshot_date')['Salary'].mean()
            else:
                ts = history_df.groupby('snapshot_date')[metric].mean()
             
            # Sorting is critical for time series
            ts = ts.sort_index()

        logger.info(f"Forecast for {metric}: Initial series size {len(ts)}, date range {ts.index.min()} to {ts.index.max()}")

        if len(ts) < 2:
            return {
                "success": False, 
                "reason": "Need at least 2 distinct data points to forecast trends."
            }
        
        # Resample to daily and fill gaps for continuity
        ts = ts.resample('D').mean().ffill().bfill()
        
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        
        # Model selection based on data points
        model = ExponentialSmoothing(
            ts, 
            trend='add' if len(ts) >= 30 else None, 
            initialization_method="estimated"
        )
        fit = model.fit()
        forecast = fit.forecast(periods * 30) # Forecast for N months (approx)
        
        # Sample down to monthly for visualization
        history_points = [{'date': d.strftime('%Y-%m-%d'), 'value': float(v)} for d, v in ts.resample('M').mean().items()]
        forecast_points = [{'date': d.strftime('%Y-%m-%d'), 'value': float(v)} for d, v in forecast.resample('M').mean().items()]
        
        return {
            "success": True,
            "metric": metric,
            "history": history_points[-24:], 
            "forecast": forecast_points[:periods]
        }
    except Exception as e:
        logger.error(f"Forecasting failed: {e}")
        return {"success": True, "success_internal": False, "reason": f"Forecasting unavailable: {str(e)}"}


@router.get("/compare-groups")
async def compare_groups(
    group_by: str = Query(..., description="Column to group by (e.g., 'Dept', 'Gender')"),
    metric: str = Query(..., description="Metric to compare (e.g., 'Salary', 'LastRating')"),
    state: AppState = Depends(require_data)
):
    """
    Compare a metric across groups using statistical hypothesis testing.
    
    Returns HR-friendly interpretation (e.g., "Confirmed Pay Gap" vs "No Significant Difference").
    """
    if state.analytics_engine is None:
        raise HTTPException(status_code=500, detail="Analytics engine not initialized")
    
    result = state.analytics_engine.compare_groups(group_by, metric)
    
    if not result.get('success'):
        raise HTTPException(status_code=400, detail=result.get('reason', 'Comparison failed'))
    
    # Translate to HR-friendly language
    if result.get('is_significant'):
        if metric.lower() == 'salary':
            hr_insight = f"⚠️ Confirmed Pay Gap: {metric} differs significantly across {group_by} groups."
        elif metric.lower() in ['lastrating', 'rating', 'performance']:
            hr_insight = f"⚠️ Performance Disparity: {metric} varies significantly by {group_by}."
        else:
            hr_insight = f"⚠️ Significant Difference: {metric} is not equal across {group_by} groups."
        
        action = "This warrants further investigation and potential intervention."
    else:
        hr_insight = f"✓ No Significant Difference in {metric} across {group_by} groups."
        action = "No immediate action required based on this comparison."
    
    return {
        **result,
        "hr_insight": hr_insight,
        "recommended_action": action
    }


