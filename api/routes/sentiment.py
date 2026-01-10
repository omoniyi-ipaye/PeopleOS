"""
API routes for Sentiment Analysis endpoints.
"""

import os
import shutil
from fastapi import APIRouter, HTTPException, Depends, Query, UploadFile, File
from fastapi.responses import FileResponse
from typing import Optional, List

from api.dependencies import get_app_state, AppState
from api.schemas.sentiment import (
    ENPSResponse,
    ENPSGroupResult,
    ENPSTrendsResponse,
    ENPSTrendPoint,
    ENPSDriversResponse,
    ENPSDriver,
    OnboardingTrajectoryResponse,
    OnboardingTrajectory,
    OnboardingTrajectorySummary,
    OnboardingHealthResponse,
    SurveyTypeMetrics,
    DimensionScore,
    EarlyWarningsResponse,
    EarlyWarning,
    EarlyWarningSummary,
    SentimentAnalysisResponse,
    SentimentSummary,
    SurveyUploadResponse,
)

router = APIRouter(prefix="/api/sentiment", tags=["sentiment"])


def require_sentiment(state: AppState = Depends(get_app_state)) -> AppState:
    """Dependency that requires sentiment engine to be available."""
    if not state.has_data():
        if not state.load_from_database():
            raise HTTPException(
                status_code=400,
                detail="No data loaded. Please upload a file first."
            )

    if state.sentiment_engine is None:
        raise HTTPException(
            status_code=400,
            detail="Sentiment analysis not available. Upload eNPS or onboarding survey data first."
        )

    return state


@router.get("/analysis", response_model=SentimentAnalysisResponse)
async def get_sentiment_analysis(
    state: AppState = Depends(require_sentiment)
) -> SentimentAnalysisResponse:
    """
    Get complete sentiment analysis.

    Includes:
    - eNPS metrics and trends
    - Onboarding trajectory analysis
    - Early warning detection
    """
    results = state.sentiment_engine.analyze_all()

    return SentimentAnalysisResponse(
        enps=results.get('enps', {}),
        enps_trends=results.get('enps_trends', {}),
        enps_drivers=results.get('enps_drivers', {}),
        onboarding=results.get('onboarding', {}),
        onboarding_health=results.get('onboarding_health', {}),
        early_warnings=results.get('early_warnings', {}),
        summary=SentimentSummary(**results.get('summary', {})),
        recommendations=results.get('recommendations', []),
        warnings=results.get('warnings', [])
    )


@router.get("/enps", response_model=ENPSResponse)
async def get_enps(
    group_by: Optional[str] = Query(
        default=None,
        description="Column to segment by (e.g., 'Dept', 'Location')"
    ),
    date_from: Optional[str] = Query(
        default=None,
        description="Start date filter (YYYY-MM-DD)"
    ),
    date_to: Optional[str] = Query(
        default=None,
        description="End date filter (YYYY-MM-DD)"
    ),
    state: AppState = Depends(require_sentiment)
) -> ENPSResponse:
    """
    Get eNPS (Employee Net Promoter Score) metrics.

    eNPS = %Promoters - %Detractors
    - Promoters: Score >= 9
    - Passives: Score 7-8
    - Detractors: Score <= 6
    """
    results = state.sentiment_engine.calculate_enps(
        group_by=group_by,
        date_from=date_from,
        date_to=date_to
    )

    if not results.get('available', False):
        return ENPSResponse(
            available=False,
            reason=results.get('reason', 'eNPS analysis not available')
        )

    by_group = []
    for g in results.get('by_group', []):
        by_group.append(ENPSGroupResult(**g))

    return ENPSResponse(
        available=True,
        overall_enps=results.get('overall_enps'),
        total_responses=results.get('total_responses'),
        promoters=results.get('promoters'),
        passives=results.get('passives'),
        detractors=results.get('detractors'),
        promoter_pct=results.get('promoter_pct'),
        passive_pct=results.get('passive_pct'),
        detractor_pct=results.get('detractor_pct'),
        interpretation=results.get('interpretation'),
        by_group=by_group
    )


@router.get("/enps/trends", response_model=ENPSTrendsResponse)
async def get_enps_trends(
    period: str = Query(
        default="month",
        description="Trend period: 'week', 'month', or 'quarter'"
    ),
    state: AppState = Depends(require_sentiment)
) -> ENPSTrendsResponse:
    """
    Get eNPS trends over time.

    Shows how eNPS has changed across periods.
    """
    results = state.sentiment_engine.get_enps_trends(period=period)

    if not results.get('available', False):
        return ENPSTrendsResponse(
            available=False,
            reason=results.get('reason', 'Trend analysis not available')
        )

    return ENPSTrendsResponse(
        available=True,
        period_type=results.get('period_type'),
        trends=[ENPSTrendPoint(**t) for t in results.get('trends', [])],
        trend_direction=results.get('trend_direction'),
        recent_change=results.get('recent_change')
    )


@router.get("/enps/drivers", response_model=ENPSDriversResponse)
async def get_enps_drivers(
    state: AppState = Depends(require_sentiment)
) -> ENPSDriversResponse:
    """
    Analyze what's driving eNPS scores.

    Identifies which dimensions (Manager, Growth, Culture, etc.)
    have the strongest correlation with overall eNPS.
    """
    results = state.sentiment_engine.get_enps_drivers()

    if not results.get('available', False):
        return ENPSDriversResponse(
            available=False,
            reason=results.get('reason', 'Driver analysis not available')
        )

    return ENPSDriversResponse(
        available=True,
        drivers=[ENPSDriver(**d) for d in results.get('drivers', [])],
        top_driver=results.get('top_driver'),
        improvement_areas=results.get('improvement_areas', []),
        recommendations=results.get('recommendations', [])
    )


@router.get("/onboarding", response_model=OnboardingTrajectoryResponse)
async def get_onboarding_trajectories(
    employee_id: Optional[str] = Query(
        default=None,
        description="Filter by specific employee ID"
    ),
    state: AppState = Depends(require_sentiment)
) -> OnboardingTrajectoryResponse:
    """
    Get onboarding survey trajectories (30/60/90 day).

    Tracks how new hire sentiment changes during onboarding period.
    """
    results = state.sentiment_engine.analyze_onboarding_trajectory(
        employee_id=employee_id
    )

    if not results.get('available', False):
        return OnboardingTrajectoryResponse(
            available=False,
            reason=results.get('reason', 'Onboarding analysis not available')
        )

    return OnboardingTrajectoryResponse(
        available=True,
        trajectories=[OnboardingTrajectory(**t) for t in results.get('trajectories', [])],
        summary=OnboardingTrajectorySummary(**results.get('summary', {})) if results.get('summary') else None,
        at_risk_employees=[OnboardingTrajectory(**t) for t in results.get('at_risk_employees', [])]
    )


@router.get("/onboarding/health", response_model=OnboardingHealthResponse)
async def get_onboarding_health(
    state: AppState = Depends(require_sentiment)
) -> OnboardingHealthResponse:
    """
    Get overall onboarding health assessment.

    Analyzes survey scores across dimensions to identify weak areas.
    """
    results = state.sentiment_engine.get_onboarding_health()

    if not results.get('available', False):
        return OnboardingHealthResponse(
            available=False,
            reason=results.get('reason', 'Onboarding health not available')
        )

    return OnboardingHealthResponse(
        available=True,
        by_survey_type=[SurveyTypeMetrics(**s) for s in results.get('by_survey_type', [])],
        dimension_scores=[DimensionScore(**d) for d in results.get('dimension_scores', [])],
        weakest_dimensions=[DimensionScore(**d) for d in results.get('weakest_dimensions', [])],
        overall_health=results.get('overall_health'),
        recommendations=results.get('recommendations', [])
    )


@router.get("/early-warnings", response_model=EarlyWarningsResponse)
async def get_early_warnings(
    state: AppState = Depends(require_sentiment)
) -> EarlyWarningsResponse:
    """
    Detect employees showing early warning signs.

    Combines eNPS and onboarding data to identify at-risk employees.
    """
    results = state.sentiment_engine.detect_early_warnings()

    if not results.get('available', False):
        return EarlyWarningsResponse(
            available=False,
            warnings=[],
            summary=None,
            recommendations=[]
        )

    return EarlyWarningsResponse(
        available=True,
        warnings=[EarlyWarning(**w) for w in results.get('warnings', [])],
        summary=EarlyWarningSummary(**results.get('summary', {})) if results.get('summary') else None,
        recommendations=results.get('recommendations', [])
    )


@router.get("/templates")
async def list_templates():
    """
    List available survey templates for download.
    """
    templates_dir = "templates"
    templates = []

    template_files = [
        {
            'name': 'onboarding_survey_template.csv',
            'description': '30/60/90 Day Onboarding Survey Template',
            'type': 'onboarding'
        },
        {
            'name': 'enps_survey_template.csv',
            'description': 'eNPS Survey Template',
            'type': 'enps'
        }
    ]

    for t in template_files:
        path = os.path.join(templates_dir, t['name'])
        if os.path.exists(path):
            templates.append({
                'name': t['name'],
                'description': t['description'],
                'type': t['type'],
                'download_url': f"/api/sentiment/templates/{t['type']}"
            })

    return {
        'templates': templates
    }


@router.get("/templates/{template_type}")
async def download_template(template_type: str):
    """
    Download a survey template.

    Args:
        template_type: 'enps' or 'onboarding'
    """
    templates_dir = "templates"

    if template_type == 'enps':
        filename = 'enps_survey_template.csv'
    elif template_type == 'onboarding':
        filename = 'onboarding_survey_template.csv'
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown template type: {template_type}. Use 'enps' or 'onboarding'."
        )

    filepath = os.path.join(templates_dir, filename)

    if not os.path.exists(filepath):
        raise HTTPException(
            status_code=404,
            detail=f"Template file not found: {filename}"
        )

    return FileResponse(
        path=filepath,
        filename=filename,
        media_type='text/csv'
    )


@router.post("/upload/enps", response_model=SurveyUploadResponse)
async def upload_enps_survey(
    file: UploadFile = File(...),
    state: AppState = Depends(get_app_state)
) -> SurveyUploadResponse:
    """
    Upload eNPS survey data.

    Expected columns:
    - EmployeeID (required)
    - SurveyDate (required)
    - eNPSScore (required, 0-10)
    - Optional: EngagementScore, WellbeingScore, GrowthScore, etc.
    """
    import pandas as pd
    import tempfile

    if not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported"
        )

    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        df = pd.read_csv(tmp_path)

        required_cols = ['EmployeeID', 'SurveyDate', 'eNPSScore']
        missing = [c for c in required_cols if c not in df.columns]

        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {missing}"
            )

        warnings = []

        # Validate score range
        if (df['eNPSScore'] < 0).any() or (df['eNPSScore'] > 10).any():
            warnings.append("Some eNPSScore values are outside 0-10 range")

        # Store in state for sentiment engine
        state.enps_df = df

        # Reinitialize sentiment engine
        from src.sentiment_engine import SentimentEngine
        state.sentiment_engine = SentimentEngine(
            employee_df=state.raw_df,
            enps_df=state.enps_df,
            onboarding_df=getattr(state, 'onboarding_df', None)
        )

        return SurveyUploadResponse(
            success=True,
            message=f"Successfully loaded {len(df)} eNPS survey responses",
            rows_loaded=len(df),
            survey_type='eNPS',
            columns_found=list(df.columns),
            warnings=warnings
        )

    finally:
        os.unlink(tmp_path)


@router.post("/upload/onboarding", response_model=SurveyUploadResponse)
async def upload_onboarding_survey(
    file: UploadFile = File(...),
    state: AppState = Depends(get_app_state)
) -> SurveyUploadResponse:
    """
    Upload onboarding survey data.

    Expected columns:
    - EmployeeID (required)
    - SurveyType (required: "30-day", "60-day", or "90-day")
    - SurveyDate (required)
    - OverallScore (required, 1-5)
    - Optional: ClarityOfRole, ManagerSupport, TeamIntegration, etc.
    """
    import pandas as pd
    import tempfile

    if not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported"
        )

    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        df = pd.read_csv(tmp_path)

        required_cols = ['EmployeeID', 'SurveyType', 'SurveyDate', 'OverallScore']
        missing = [c for c in required_cols if c not in df.columns]

        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {missing}"
            )

        warnings = []

        # Validate survey types
        valid_types = ['30-day', '60-day', '90-day']
        invalid_types = set(df['SurveyType'].unique()) - set(valid_types)
        if invalid_types:
            warnings.append(f"Unknown survey types found: {invalid_types}")

        # Store in state for sentiment engine
        state.onboarding_df = df

        # Reinitialize sentiment engine
        from src.sentiment_engine import SentimentEngine
        state.sentiment_engine = SentimentEngine(
            employee_df=state.raw_df,
            enps_df=getattr(state, 'enps_df', None),
            onboarding_df=state.onboarding_df
        )

        return SurveyUploadResponse(
            success=True,
            message=f"Successfully loaded {len(df)} onboarding survey responses",
            rows_loaded=len(df),
            survey_type='Onboarding',
            columns_found=list(df.columns),
            warnings=warnings
        )

    finally:
        os.unlink(tmp_path)
