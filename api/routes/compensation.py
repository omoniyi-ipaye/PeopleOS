"""Governed aggregate compensation routes.

Current compensation analytics use the active workforce. Department dispersion is
a descriptive consistency metric, not adjusted pay equity. Individual salary
outlier and employee compa-ratio lists are outside the enterprise aggregate
boundary and are disabled.
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.dependencies import AppState, get_app_state

from src.serialization import json_safe

router = APIRouter(prefix="/api/compensation", tags=["compensation"])


class CompensationSummary(BaseModel):
    total_payroll: float
    avg_salary: float
    median_salary: float
    min_salary: float
    max_salary: float
    salary_range: float
    std_dev: float
    headcount: int
    active_count: Optional[int] = None
    salary_observations: Optional[int] = None
    excluded_salary_count: Optional[int] = None
    salary_coverage: Optional[float] = None
    population: str = 'current_active_employees_with_valid_positive_salary'


class SalaryDispersionScore(BaseModel):
    dept: str
    avg_salary: float
    std_dev: float
    cv: float
    gini: float
    equity_score: float  # compatibility field: salary-dispersion consistency score
    status: str
    headcount: int
    metric_semantics: str = "salary_dispersion_consistency_not_adjusted_pay_equity"


class CompensationAnalysisResponse(BaseModel):
    summary: CompensationSummary
    equity_scores: List[SalaryDispersionScore]
    outliers: List[Dict[str, Any]] = []
    warnings: List[str]


def require_compensation(state: AppState = Depends(get_app_state)) -> AppState:
    if not state.has_data():
        if not state.load_from_database():
            raise HTTPException(status_code=400, detail="No data loaded. Please upload a file first.")
    if state.compensation_engine is None:
        raise HTTPException(status_code=400, detail="Compensation analysis is unavailable for the current dataset.")
    return state


def _summary(engine) -> CompensationSummary:
    raw = engine.get_compensation_summary()
    return CompensationSummary(
        total_payroll=raw['total_payroll'],
        avg_salary=raw['avg_salary'],
        median_salary=raw['median_salary'],
        min_salary=raw['min_salary'],
        max_salary=raw['max_salary'],
        salary_range=raw['salary_range'],
        std_dev=raw['std_dev'],
        headcount=raw['headcount'],
        active_count=raw.get('active_count'),
        salary_observations=raw.get('salary_observations'),
        excluded_salary_count=raw.get('excluded_salary_count'),
        salary_coverage=raw.get('salary_coverage'),
    )


def _dispersion(engine) -> List[SalaryDispersionScore]:
    frame = engine.calculate_pay_equity_score()
    if not frame.empty and 'Headcount' in frame:
        frame = frame[frame['Headcount'] >= 10]
    output = []
    for _, row in frame.iterrows():
        output.append(SalaryDispersionScore(
            dept=row['Dept'],
            avg_salary=float(row['AvgSalary']),
            std_dev=float(row['StdDev']),
            cv=float(row['CV']),
            gini=float(row['Gini']),
            equity_score=float(row.get('SalaryDispersionScore', row.get('EquityScore'))),
            status=row['Status'],
            headcount=int(row['Headcount']),
        ))
    return output


@router.get("/summary", response_model=CompensationSummary)
async def get_compensation_summary(state: AppState = Depends(require_compensation)) -> CompensationSummary:
    return _summary(state.compensation_engine)


@router.get("/equity", response_model=List[SalaryDispersionScore])
async def get_salary_dispersion(state: AppState = Depends(require_compensation)) -> List[SalaryDispersionScore]:
    """Compatibility endpoint: returns department salary-dispersion screening, not a legal/adjusted pay-equity determination."""
    return _dispersion(state.compensation_engine)


@router.get("/outliers", deprecated=True)
async def get_salary_outliers(state: AppState = Depends(require_compensation)):
    raise HTTPException(status_code=403, detail="Individual salary-outlier lists are disabled. Use aggregate compensation disparity analysis.")


@router.get("/compa-ratio", deprecated=True)
async def get_compa_ratios(state: AppState = Depends(require_compensation)):
    raise HTTPException(status_code=403, detail="Individual compa-ratio lists are disabled. Use aggregate compensation analysis.")


@router.get("/gender-pay-gap")
async def get_gender_pay_gap(state: AppState = Depends(require_compensation)) -> Dict[str, Any]:
    result = state.compensation_engine.calculate_gender_pay_gap()
    if isinstance(result, dict):
        result.setdefault('metric_semantics', 'descriptive_pay_gap_screening_not_legal_equity_determination')
    return result


@router.get("/by-tenure")
async def get_salary_by_tenure(state: AppState = Depends(require_compensation)) -> List[Dict[str, Any]]:
    frame = state.compensation_engine.get_salary_by_tenure()
    if not frame.empty and 'Count' in frame:
        frame = frame[frame['Count'] >= 10]
    return [
        {
            'tenure_bucket': str(row['TenureBucket']),
            'mean': json_safe(row['Mean']),
            'median': json_safe(row['Median']),
            'min': json_safe(row['Min']),
            'max': json_safe(row['Max']),
            'count': int(row['Count']),
            'population': 'current_active_employees_with_valid_salary',
        }
        for _, row in frame.iterrows()
    ]


@router.get("/analysis", response_model=CompensationAnalysisResponse)
async def get_full_analysis(state: AppState = Depends(require_compensation)) -> CompensationAnalysisResponse:
    engine = state.compensation_engine
    return CompensationAnalysisResponse(
        summary=_summary(engine),
        equity_scores=_dispersion(engine),
        # Enterprise boundary intentionally suppresses individual salary records.
        outliers=[],
        warnings=list(getattr(engine, 'warnings', []) or []) + [
            "Department consistency scores describe salary dispersion; they are not adjusted pay-equity findings.",
            "Individual salary-outlier and compa-ratio rankings are not exposed through the governed aggregate API.",
        ],
    )
