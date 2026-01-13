"""Team dynamics route handlers."""

from typing import List, Dict, Any, Optional

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel

from api.dependencies import get_app_state, AppState
import pandas as pd
import numpy as np

router = APIRouter(prefix="/api/team", tags=["team"])


class TeamHealth(BaseModel):
    """Team health score."""
    dept: str
    health_score: float
    avg_tenure: float | None
    avg_rating: float | None
    headcount: int
    attrition_rate: float | None
    status: str


class DiversityMetrics(BaseModel):
    """Diversity metrics for a department."""
    dept: str
    headcount: int
    tenure_diversity: float | None
    age_diversity: float | None
    salary_equity: float | None
    overall_diversity: float


class GenderBreakdown(BaseModel):
    """Gender distribution."""
    male: int
    female: int
    other: int
    male_pct: float
    female_pct: float
    other_pct: float


class AgeDistribution(BaseModel):
    """Age group distribution."""
    group: str
    count: int
    percentage: float


class SatisfactionMetrics(BaseModel):
    """Team satisfaction metrics."""
    avg_enps: float | None
    avg_pulse: float | None
    avg_manager_satisfaction: float | None
    avg_work_life_balance: float | None
    avg_career_growth: float | None


class PerformanceDistribution(BaseModel):
    """Performance rating distribution."""
    rating_range: str
    count: int
    percentage: float


class TeamComposition(BaseModel):
    """Team composition breakdown."""
    by_tenure: List[Dict[str, Any]]
    by_job_level: List[Dict[str, Any]]
    avg_span_of_control: float | None


class ComprehensiveTeamMetrics(BaseModel):
    """Complete team dynamics metrics with filters applied."""
    total_employees: int
    filters_applied: Dict[str, Any]

    # Demographics
    gender_breakdown: GenderBreakdown
    age_distribution: List[AgeDistribution]
    location_distribution: List[Dict[str, Any]]

    # Composition
    composition: TeamComposition

    # Performance
    avg_rating: float | None
    performance_distribution: List[PerformanceDistribution]
    top_performers_count: int
    top_performers_pct: float

    # Satisfaction
    satisfaction: SatisfactionMetrics

    # Stability
    avg_tenure: float | None
    attrition_rate: float | None
    avg_manager_changes: float | None
    new_hires_count: int  # < 1 year
    veterans_count: int   # > 5 years

    # Department breakdown (for comparison)
    by_department: List[Dict[str, Any]]


class FilterOptions(BaseModel):
    """Available filter options."""
    departments: List[str]
    locations: List[str]
    countries: List[str]
    genders: List[str]
    age_groups: List[str]
    job_levels: List[int]
    tenure_ranges: List[str]


def require_team(state: AppState = Depends(get_app_state)) -> AppState:
    """Dependency that requires team dynamics engine."""
    if not state.has_data():
        if not state.load_from_database():
            raise HTTPException(
                status_code=400,
                detail="No data loaded. Please upload a file first."
            )

    if state.team_dynamics_engine is None:
        raise HTTPException(
            status_code=500,
            detail="Team dynamics engine not initialized"
        )

    return state


@router.get("/health", response_model=List[TeamHealth])
async def get_team_health(
    state: AppState = Depends(require_team)
) -> List[TeamHealth]:
    """
    Get health scores for all teams/departments.
    """
    health_df = state.team_dynamics_engine.calculate_team_health_scores()

    teams = []
    for _, row in health_df.iterrows():
        teams.append(TeamHealth(
            dept=row['Dept'],
            health_score=float(row['HealthScore']),
            avg_tenure=float(row['AvgTenure']) if row.get('AvgTenure') is not None else None,
            avg_rating=float(row['AvgRating']) if row.get('AvgRating') is not None else None,
            headcount=int(row['Headcount']),
            attrition_rate=float(row['AttritionRate']) if row.get('AttritionRate') is not None else None,
            status=row['Status']
        ))

    return teams


@router.get("/diversity", response_model=List[DiversityMetrics])
async def get_diversity_metrics(
    state: AppState = Depends(require_team)
) -> List[DiversityMetrics]:
    """
    Get diversity metrics for all teams/departments.
    """
    diversity_df = state.team_dynamics_engine.analyze_team_diversity()

    metrics = []
    for _, row in diversity_df.iterrows():
        metrics.append(DiversityMetrics(
            dept=row['Dept'],
            headcount=int(row['Headcount']),
            tenure_diversity=float(row['TenureDiversity']) if 'TenureDiversity' in row else None,
            age_diversity=float(row['AgeDiversity']) if 'AgeDiversity' in row else None,
            salary_equity=float(row['SalaryEquity']) if 'SalaryEquity' in row else None,
            overall_diversity=float(row['OverallDiversity'])
        ))

    return metrics


@router.get("/analysis")
async def get_team_analysis(
    state: AppState = Depends(require_team)
) -> Dict[str, Any]:
    """
    Get full team dynamics analysis.
    """
    analysis = state.team_dynamics_engine.analyze_all()

    # Convert DataFrames to lists of dicts
    result = {
        'health': [],
        'diversity': [],
        'at_risk_teams': [],
        'summary': analysis.get('summary', {})
    }

    if 'health' in analysis and not analysis['health'].empty:
        for _, row in analysis['health'].iterrows():
            result['health'].append({
                'dept': row['Dept'],
                'health_score': float(row['HealthScore']),
                'avg_tenure': float(row['AvgTenure']) if row.get('AvgTenure') is not None else None,
                'avg_rating': float(row['AvgRating']) if row.get('AvgRating') is not None else None,
                'headcount': int(row['Headcount']),
                'attrition_rate': float(row['AttritionRate']) if row.get('AttritionRate') is not None else None,
                'status': row['Status']
            })

    if 'diversity' in analysis and not analysis['diversity'].empty:
        for _, row in analysis['diversity'].iterrows():
            result['diversity'].append({
                'dept': row['Dept'],
                'headcount': int(row['Headcount']),
                'tenure_diversity': float(row['TenureDiversity']) if 'TenureDiversity' in row else None,
                'age_diversity': float(row['AgeDiversity']) if 'AgeDiversity' in row else None,
                'salary_equity': float(row['SalaryEquity']) if 'SalaryEquity' in row else None,
                'overall_diversity': float(row['OverallDiversity'])
            })

    if 'at_risk' in analysis and not analysis['at_risk'].empty:
        for _, row in analysis['at_risk'].iterrows():
            result['at_risk_teams'].append({
                'dept': row['Dept'],
                'headcount': int(row['Headcount']),
                'health_score': float(row['HealthScore']),
                'status': row['Status'],
                'risk_factors': row.get('RiskFactors', ''),
                'recommendations': row.get('Recommendations', '')
            })

    return result


def require_data(state: AppState = Depends(get_app_state)) -> AppState:
    """Dependency that requires data loaded."""
    if not state.has_data():
        if not state.load_from_database():
            raise HTTPException(
                status_code=400,
                detail="No data loaded. Please upload a file first."
            )
    return state


def get_age_group(age: float) -> str:
    """Convert age to generation group."""
    if pd.isna(age):
        return "Unknown"
    if age < 25:
        return "Gen Z (< 25)"
    elif age < 40:
        return "Millennial (25-39)"
    elif age < 55:
        return "Gen X (40-54)"
    else:
        return "Boomer (55+)"


def get_tenure_group(tenure: float) -> str:
    """Convert tenure to group."""
    if pd.isna(tenure):
        return "Unknown"
    if tenure < 1:
        return "New (< 1 yr)"
    elif tenure < 3:
        return "Growing (1-3 yrs)"
    elif tenure < 5:
        return "Established (3-5 yrs)"
    else:
        return "Veteran (5+ yrs)"


@router.get("/filters", response_model=FilterOptions)
async def get_filter_options(
    state: AppState = Depends(require_data)
) -> FilterOptions:
    """Get available filter options based on current data."""
    df = state.raw_df

    # Get unique values for each filterable field
    departments = sorted(df['Dept'].dropna().unique().tolist()) if 'Dept' in df.columns else []
    locations = sorted(df['Location'].dropna().unique().tolist()) if 'Location' in df.columns else []
    countries = sorted(df['Country'].dropna().unique().tolist()) if 'Country' in df.columns else []
    genders = sorted(df['Gender'].dropna().unique().tolist()) if 'Gender' in df.columns else []

    # Age groups
    if 'Age' in df.columns:
        age_groups = ["Gen Z (< 25)", "Millennial (25-39)", "Gen X (40-54)", "Boomer (55+)"]
    else:
        age_groups = []

    # Job levels
    if 'JobLevel' in df.columns:
        job_levels = sorted([int(x) for x in df['JobLevel'].dropna().unique().tolist()])
    else:
        job_levels = []

    # Tenure ranges
    tenure_ranges = ["New (< 1 yr)", "Growing (1-3 yrs)", "Established (3-5 yrs)", "Veteran (5+ yrs)"]

    return FilterOptions(
        departments=departments,
        locations=locations,
        countries=countries,
        genders=genders,
        age_groups=age_groups,
        job_levels=job_levels,
        tenure_ranges=tenure_ranges
    )


@router.get("/comprehensive", response_model=ComprehensiveTeamMetrics)
async def get_comprehensive_metrics(
    state: AppState = Depends(require_data),
    departments: Optional[str] = Query(None, description="Comma-separated department names"),
    locations: Optional[str] = Query(None, description="Comma-separated locations"),
    countries: Optional[str] = Query(None, description="Comma-separated countries"),
    genders: Optional[str] = Query(None, description="Comma-separated genders"),
    age_groups: Optional[str] = Query(None, description="Comma-separated age groups"),
    job_levels: Optional[str] = Query(None, description="Comma-separated job levels"),
    tenure_ranges: Optional[str] = Query(None, description="Comma-separated tenure ranges"),
    min_tenure: Optional[float] = Query(None, description="Minimum tenure"),
    max_tenure: Optional[float] = Query(None, description="Maximum tenure"),
) -> ComprehensiveTeamMetrics:
    """
    Get comprehensive team dynamics metrics with real-time filtering.
    All filters are applied server-side for fast response.
    """
    df = state.raw_df.copy()
    filters_applied = {}

    # Apply filters
    if departments:
        dept_list = [d.strip() for d in departments.split(",")]
        df = df[df['Dept'].isin(dept_list)]
        filters_applied['departments'] = dept_list

    if locations and 'Location' in df.columns:
        loc_list = [l.strip() for l in locations.split(",")]
        df = df[df['Location'].isin(loc_list)]
        filters_applied['locations'] = loc_list

    if countries and 'Country' in df.columns:
        country_list = [c.strip() for c in countries.split(",")]
        df = df[df['Country'].isin(country_list)]
        filters_applied['countries'] = country_list

    if genders and 'Gender' in df.columns:
        gender_list = [g.strip() for g in genders.split(",")]
        df = df[df['Gender'].isin(gender_list)]
        filters_applied['genders'] = gender_list

    if age_groups and 'Age' in df.columns:
        age_group_list = [a.strip() for a in age_groups.split(",")]
        df['_age_group'] = df['Age'].apply(get_age_group)
        df = df[df['_age_group'].isin(age_group_list)]
        filters_applied['age_groups'] = age_group_list

    if job_levels and 'JobLevel' in df.columns:
        level_list = [int(l.strip()) for l in job_levels.split(",")]
        df = df[df['JobLevel'].isin(level_list)]
        filters_applied['job_levels'] = level_list

    if tenure_ranges and 'Tenure' in df.columns:
        tenure_list = [t.strip() for t in tenure_ranges.split(",")]
        df['_tenure_group'] = df['Tenure'].apply(get_tenure_group)
        df = df[df['_tenure_group'].isin(tenure_list)]
        filters_applied['tenure_ranges'] = tenure_list

    if min_tenure is not None and 'Tenure' in df.columns:
        df = df[df['Tenure'] >= min_tenure]
        filters_applied['min_tenure'] = min_tenure

    if max_tenure is not None and 'Tenure' in df.columns:
        df = df[df['Tenure'] <= max_tenure]
        filters_applied['max_tenure'] = max_tenure

    total = len(df)
    if total == 0:
        # Return empty metrics
        return ComprehensiveTeamMetrics(
            total_employees=0,
            filters_applied=filters_applied,
            gender_breakdown=GenderBreakdown(male=0, female=0, other=0, male_pct=0, female_pct=0, other_pct=0),
            age_distribution=[],
            location_distribution=[],
            composition=TeamComposition(by_tenure=[], by_job_level=[], avg_span_of_control=None),
            avg_rating=None,
            performance_distribution=[],
            top_performers_count=0,
            top_performers_pct=0,
            satisfaction=SatisfactionMetrics(
                avg_enps=None, avg_pulse=None, avg_manager_satisfaction=None,
                avg_work_life_balance=None, avg_career_growth=None
            ),
            avg_tenure=None,
            attrition_rate=None,
            avg_manager_changes=None,
            new_hires_count=0,
            veterans_count=0,
            by_department=[]
        )

    # Gender breakdown
    if 'Gender' in df.columns:
        gender_counts = df['Gender'].value_counts()
        male = int(gender_counts.get('Male', 0))
        female = int(gender_counts.get('Female', 0))
        other = total - male - female
        gender_breakdown = GenderBreakdown(
            male=male,
            female=female,
            other=max(0, other),
            male_pct=round(male / total * 100, 1) if total > 0 else 0,
            female_pct=round(female / total * 100, 1) if total > 0 else 0,
            other_pct=round(other / total * 100, 1) if total > 0 and other > 0 else 0
        )
    else:
        gender_breakdown = GenderBreakdown(male=0, female=0, other=total, male_pct=0, female_pct=0, other_pct=100)

    # Age distribution
    age_distribution = []
    if 'Age' in df.columns:
        df['_age_group_calc'] = df['Age'].apply(get_age_group)
        age_counts = df['_age_group_calc'].value_counts()
        for group in ["Gen Z (< 25)", "Millennial (25-39)", "Gen X (40-54)", "Boomer (55+)"]:
            count = int(age_counts.get(group, 0))
            age_distribution.append(AgeDistribution(
                group=group,
                count=count,
                percentage=round(count / total * 100, 1) if total > 0 else 0
            ))

    # Location distribution
    location_distribution = []
    if 'Location' in df.columns:
        loc_counts = df['Location'].value_counts().head(10)
        for loc, count in loc_counts.items():
            location_distribution.append({
                'location': loc,
                'count': int(count),
                'percentage': round(count / total * 100, 1)
            })

    # Tenure composition
    tenure_composition = []
    if 'Tenure' in df.columns:
        df['_tenure_group_calc'] = df['Tenure'].apply(get_tenure_group)
        tenure_counts = df['_tenure_group_calc'].value_counts()
        for group in ["New (< 1 yr)", "Growing (1-3 yrs)", "Established (3-5 yrs)", "Veteran (5+ yrs)"]:
            count = int(tenure_counts.get(group, 0))
            tenure_composition.append({
                'group': group,
                'count': count,
                'percentage': round(count / total * 100, 1) if total > 0 else 0
            })

    # Job level composition
    job_level_composition = []
    if 'JobLevel' in df.columns:
        level_counts = df['JobLevel'].value_counts().sort_index()
        level_names = {1: "Entry", 2: "Junior", 3: "Mid", 4: "Senior", 5: "Lead", 6: "Director", 7: "VP", 8: "Executive"}
        for level, count in level_counts.items():
            job_level_composition.append({
                'level': int(level),
                'name': level_names.get(int(level), f"Level {int(level)}"),
                'count': int(count),
                'percentage': round(count / total * 100, 1) if total > 0 else 0
            })

    # Span of control (reports per manager)
    avg_span = None
    if 'ManagerID' in df.columns:
        manager_counts = df['ManagerID'].value_counts()
        if len(manager_counts) > 0:
            avg_span = round(manager_counts.mean(), 1)

    composition = TeamComposition(
        by_tenure=tenure_composition,
        by_job_level=job_level_composition,
        avg_span_of_control=avg_span
    )

    # Performance metrics
    avg_rating = None
    performance_distribution = []
    top_performers_count = 0
    if 'LastRating' in df.columns:
        avg_rating = round(df['LastRating'].mean(), 2) if not df['LastRating'].isna().all() else None

        # Distribution by rating ranges
        rating_ranges = [
            ("Outstanding (4.5+)", df['LastRating'] >= 4.5),
            ("Exceeds (4.0-4.4)", (df['LastRating'] >= 4.0) & (df['LastRating'] < 4.5)),
            ("Meets (3.0-3.9)", (df['LastRating'] >= 3.0) & (df['LastRating'] < 4.0)),
            ("Below (2.0-2.9)", (df['LastRating'] >= 2.0) & (df['LastRating'] < 3.0)),
            ("Needs Improvement (< 2.0)", df['LastRating'] < 2.0),
        ]
        for range_name, mask in rating_ranges:
            count = int(mask.sum())
            performance_distribution.append(PerformanceDistribution(
                rating_range=range_name,
                count=count,
                percentage=round(count / total * 100, 1) if total > 0 else 0
            ))

        top_performers_count = int((df['LastRating'] >= 4.0).sum())

    top_performers_pct = round(top_performers_count / total * 100, 1) if total > 0 else 0

    # Satisfaction metrics
    def safe_mean(col):
        if col in df.columns and not df[col].isna().all():
            return round(df[col].mean(), 2)
        return None

    satisfaction = SatisfactionMetrics(
        avg_enps=safe_mean('eNPS_Score'),
        avg_pulse=safe_mean('Pulse_Score'),
        avg_manager_satisfaction=safe_mean('ManagerSatisfaction'),
        avg_work_life_balance=safe_mean('WorkLifeBalance'),
        avg_career_growth=safe_mean('CareerGrowthSatisfaction')
    )

    # Stability metrics
    avg_tenure = round(df['Tenure'].mean(), 2) if 'Tenure' in df.columns and not df['Tenure'].isna().all() else None

    attrition_rate = None
    if 'Attrition' in df.columns:
        attrition_rate = round(df['Attrition'].mean() * 100, 1)

    avg_manager_changes = safe_mean('ManagerChangeCount')

    new_hires_count = int((df['Tenure'] < 1).sum()) if 'Tenure' in df.columns else 0
    veterans_count = int((df['Tenure'] >= 5).sum()) if 'Tenure' in df.columns else 0

    # Department breakdown
    by_department = []
    if 'Dept' in df.columns:
        for dept in df['Dept'].unique():
            dept_df = df[df['Dept'] == dept]
            dept_total = len(dept_df)

            dept_data = {
                'department': dept,
                'headcount': dept_total,
                'percentage': round(dept_total / total * 100, 1) if total > 0 else 0,
            }

            # Gender ratio for department
            if 'Gender' in dept_df.columns:
                male_count = (dept_df['Gender'] == 'Male').sum()
                female_count = (dept_df['Gender'] == 'Female').sum()
                dept_data['male_pct'] = round(male_count / dept_total * 100, 1) if dept_total > 0 else 0
                dept_data['female_pct'] = round(female_count / dept_total * 100, 1) if dept_total > 0 else 0

            # Avg rating
            if 'LastRating' in dept_df.columns and not dept_df['LastRating'].isna().all():
                dept_data['avg_rating'] = round(dept_df['LastRating'].mean(), 2)

            # Avg tenure
            if 'Tenure' in dept_df.columns and not dept_df['Tenure'].isna().all():
                dept_data['avg_tenure'] = round(dept_df['Tenure'].mean(), 1)

            # Satisfaction scores
            if 'eNPS_Score' in dept_df.columns and not dept_df['eNPS_Score'].isna().all():
                dept_data['avg_enps'] = round(dept_df['eNPS_Score'].mean(), 1)

            if 'ManagerSatisfaction' in dept_df.columns and not dept_df['ManagerSatisfaction'].isna().all():
                dept_data['avg_manager_satisfaction'] = round(dept_df['ManagerSatisfaction'].mean(), 2)

            by_department.append(dept_data)

        # Sort by headcount
        by_department.sort(key=lambda x: x['headcount'], reverse=True)

    return ComprehensiveTeamMetrics(
        total_employees=total,
        filters_applied=filters_applied,
        gender_breakdown=gender_breakdown,
        age_distribution=age_distribution,
        location_distribution=location_distribution,
        composition=composition,
        avg_rating=avg_rating,
        performance_distribution=performance_distribution,
        top_performers_count=top_performers_count,
        top_performers_pct=top_performers_pct,
        satisfaction=satisfaction,
        avg_tenure=avg_tenure,
        attrition_rate=attrition_rate,
        avg_manager_changes=avg_manager_changes,
        new_hires_count=new_hires_count,
        veterans_count=veterans_count,
        by_department=by_department
    )
