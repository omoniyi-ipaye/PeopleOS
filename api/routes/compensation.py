"""Compensation analysis route handlers."""

from typing import List, Dict, Any

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from api.dependencies import get_app_state, AppState

router = APIRouter(prefix="/api/compensation", tags=["compensation"])


class CompensationSummary(BaseModel):
    """Compensation summary statistics."""
    total_payroll: float
    avg_salary: float
    median_salary: float
    min_salary: float
    max_salary: float
    salary_range: float
    std_dev: float
    headcount: int


class PayEquityScore(BaseModel):
    """Pay equity score for a department."""
    dept: str
    avg_salary: float
    std_dev: float
    cv: float
    gini: float
    equity_score: float
    status: str
    headcount: int


class SalaryOutlier(BaseModel):
    """Salary outlier information."""
    employee_id: str
    dept: str
    salary: float
    dept_avg: float
    deviation_pct: float
    z_score: float
    flag: str


class CompaRatioEmployee(BaseModel):
    """Compa-ratio for an employee."""
    employee_id: str
    dept: str
    salary: float
    band_midpoint: float
    compa_ratio: float
    compa_status: str


class CompensationAnalysisResponse(BaseModel):
    """Full compensation analysis response."""
    summary: CompensationSummary
    equity_scores: List[PayEquityScore]
    outliers: List[SalaryOutlier]
    warnings: List[str]


def require_compensation(state: AppState = Depends(get_app_state)) -> AppState:
    """Dependency that requires compensation engine."""
    if not state.has_data():
        if not state.load_from_database():
            raise HTTPException(
                status_code=400,
                detail="No data loaded. Please upload a file first."
            )

    if state.compensation_engine is None:
        raise HTTPException(
            status_code=500,
            detail="Compensation engine not initialized"
        )

    return state


@router.get("/summary", response_model=CompensationSummary)
async def get_compensation_summary(
    state: AppState = Depends(require_compensation)
) -> CompensationSummary:
    """
    Get overall compensation summary statistics.
    """
    summary = state.compensation_engine.get_compensation_summary()

    return CompensationSummary(
        total_payroll=summary['total_payroll'],
        avg_salary=summary['avg_salary'],
        median_salary=summary['median_salary'],
        min_salary=summary['min_salary'],
        max_salary=summary['max_salary'],
        salary_range=summary['salary_range'],
        std_dev=summary['std_dev'],
        headcount=summary['headcount']
    )


@router.get("/equity", response_model=List[PayEquityScore])
async def get_pay_equity(
    state: AppState = Depends(require_compensation)
) -> List[PayEquityScore]:
    """
    Get pay equity scores by department.
    """
    equity_df = state.compensation_engine.calculate_pay_equity_score()

    scores = []
    for _, row in equity_df.iterrows():
        scores.append(PayEquityScore(
            dept=row['Dept'],
            avg_salary=float(row['AvgSalary']),
            std_dev=float(row['StdDev']),
            cv=float(row['CV']),
            gini=float(row['Gini']),
            equity_score=float(row['EquityScore']),
            status=row['Status'],
            headcount=int(row['Headcount'])
        ))

    return scores


@router.get("/outliers", response_model=List[SalaryOutlier])
async def get_salary_outliers(
    state: AppState = Depends(require_compensation)
) -> List[SalaryOutlier]:
    """
    Get employees with outlier salaries.
    """
    outliers_df = state.compensation_engine.identify_salary_outliers()

    outliers = []
    for _, row in outliers_df.iterrows():
        outliers.append(SalaryOutlier(
            employee_id=row['EmployeeID'],
            dept=row['Dept'],
            salary=float(row['Salary']),
            dept_avg=float(row['DeptAvg']),
            deviation_pct=float(row['DeviationPct']),
            z_score=float(row['ZScore']),
            flag=row['Flag']
        ))

    return outliers


@router.get("/compa-ratio", response_model=List[CompaRatioEmployee])
async def get_compa_ratios(
    state: AppState = Depends(require_compensation)
) -> List[CompaRatioEmployee]:
    """
    Get compa-ratio analysis for all employees.
    """
    compa_df = state.compensation_engine.calculate_compa_ratio()

    ratios = []
    for _, row in compa_df.iterrows():
        ratios.append(CompaRatioEmployee(
            employee_id=row['EmployeeID'],
            dept=row.get('Dept', 'N/A'),
            salary=float(row['Salary']),
            band_midpoint=float(row['BandMidpoint']),
            compa_ratio=float(row['CompaRatio']),
            compa_status=row['CompaStatus']
        ))

    return ratios


@router.get("/gender-pay-gap")
async def get_gender_pay_gap(
    state: AppState = Depends(require_compensation)
) -> Dict[str, Any]:
    """
    Get gender pay gap analysis (requires Gender column).
    """
    result = state.compensation_engine.calculate_gender_pay_gap()
    return result


@router.get("/by-tenure")
async def get_salary_by_tenure(
    state: AppState = Depends(require_compensation)
) -> List[Dict[str, Any]]:
    """
    Get salary analysis by tenure bucket.
    """
    tenure_df = state.compensation_engine.get_salary_by_tenure()

    results = []
    for _, row in tenure_df.iterrows():
        results.append({
            'tenure_bucket': str(row['TenureBucket']),
            'mean': float(row['Mean']),
            'median': float(row['Median']),
            'min': float(row['Min']),
            'max': float(row['Max']),
            'count': int(row['Count'])
        })

    return results


@router.get("/analysis", response_model=CompensationAnalysisResponse)
async def get_full_analysis(
    state: AppState = Depends(require_compensation)
) -> CompensationAnalysisResponse:
    """
    Get full compensation analysis.
    """
    analysis = state.compensation_engine.analyze_all()

    # Convert summary
    summary = CompensationSummary(
        total_payroll=analysis['summary']['total_payroll'],
        avg_salary=analysis['summary']['avg_salary'],
        median_salary=analysis['summary']['median_salary'],
        min_salary=analysis['summary']['min_salary'],
        max_salary=analysis['summary']['max_salary'],
        salary_range=analysis['summary']['salary_range'],
        std_dev=analysis['summary']['std_dev'],
        headcount=analysis['summary']['headcount']
    )

    # Convert equity scores
    equity_scores = []
    for _, row in analysis['equity'].iterrows():
        equity_scores.append(PayEquityScore(
            dept=row['Dept'],
            avg_salary=float(row['AvgSalary']),
            std_dev=float(row['StdDev']),
            cv=float(row['CV']),
            gini=float(row['Gini']),
            equity_score=float(row['EquityScore']),
            status=row['Status'],
            headcount=int(row['Headcount'])
        ))

    # Convert outliers
    outliers = []
    for _, row in analysis['outliers'].iterrows():
        outliers.append(SalaryOutlier(
            employee_id=row['EmployeeID'],
            dept=row['Dept'],
            salary=float(row['Salary']),
            dept_avg=float(row['DeptAvg']),
            deviation_pct=float(row['DeviationPct']),
            z_score=float(row['ZScore']),
            flag=row['Flag']
        ))

    return CompensationAnalysisResponse(
        summary=summary,
        equity_scores=equity_scores,
        outliers=outliers,
        warnings=analysis.get('warnings', [])
    )
