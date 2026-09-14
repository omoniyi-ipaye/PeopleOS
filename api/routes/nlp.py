"""NLP analysis route handlers."""

from typing import List, Dict, Any, Optional

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field, ValidationError

from src.platform.provenance import IntegrityError, snapshot_provenance
from src.platform.runtime_lock import RUNTIME_MUTATION_LOCK
from api.dependencies import get_app_state, AppState

router = APIRouter(prefix="/api/nlp", tags=["nlp"])


class SentimentSummary(BaseModel):
    """Sentiment analysis summary statistics."""
    avg_sentiment: Optional[float] = None
    positive_count: int
    neutral_count: int
    negative_count: int
    positive_pct: float
    neutral_pct: float
    negative_pct: float
    sentiment_observations: int = 0
    excluded_sentiment_rows: int = 0
    unprocessed_sentiment_rows: int = 0


class TopicInfo(BaseModel):
    """NLP topic information."""
    name: str
    description: str
    prevalence: Optional[str] = None
    measurement_semantics: str = "generated_theme_not_measured_prevalence"
    sample_size: Optional[int] = None
    sample_scope: Optional[str] = None
    sentiment: Optional[str] = None
    count: Optional[int] = None


class NLPAnalysisResponse(BaseModel):
    """Full NLP analysis response."""
    provenance: Optional[Dict[str, Any]] = None
    sentiment_summary: SentimentSummary
    topics: List[TopicInfo]
    skills: Dict[str, Any]
    nlp_available: bool
    analysis_status: str = 'unavailable'
    component_status: Dict[str, Any] = Field(default_factory=dict)
    unavailable_reason: Optional[str] = None


def require_nlp(state: AppState = Depends(get_app_state)) -> AppState:
    """Dependency that requires NLP engine."""
    if not state.has_data():
        if not state.load_from_database():
            raise HTTPException(
                status_code=400,
                detail="No data loaded. Please upload a file first."
            )

    if state.nlp_engine is None:
        raise HTTPException(
            status_code=503,
            detail="NLP engine not initialized"
        )

    return state


@router.get("/analysis", response_model=NLPAnalysisResponse)
def get_nlp_analysis(
    state: AppState = Depends(require_nlp)
) -> NLPAnalysisResponse:
    """
    Get full NLP analysis of performance reviews.
    """
    try:
        with RUNTIME_MUTATION_LOCK:
            provenance = snapshot_provenance(state)
            if 'PerformanceText' not in state.raw_df.columns:
                raise HTTPException(status_code=400, detail='No PerformanceText column found in dataset')
            if state.nlp_results is not None and state.nlp_results.get('provenance') == provenance:
                try:
                    return NLPAnalysisResponse(**state.nlp_results)
                except ValidationError as exc:
                    state.nlp_results = None
                    raise HTTPException(
                        status_code=409,
                        detail='Cached NLP analysis failed integrity validation. Retry the analysis.',
                    ) from exc
            engine = state.nlp_engine
            if engine is None:
                raise HTTPException(status_code=503, detail='NLP engine not initialized')
            source = state.raw_df.copy(deep=True)
    except IntegrityError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    # Inference may take minutes; it must not block dataset/reset mutations.
    try:
        analysis = engine.process_all(source)
    except Exception as exc:
        raise HTTPException(status_code=503, detail='NLP analysis unavailable; no result was cached.') from exc
    if not isinstance(analysis, dict):
        raise HTTPException(status_code=503, detail='NLP analysis returned no governed result; no result was cached.')
    try:
        with RUNTIME_MUTATION_LOCK:
            if snapshot_provenance(state) != provenance:
                raise IntegrityError('Dataset changed while text analysis was running. Retry the analysis.')
            try:
                response = NLPAnalysisResponse(
                    provenance=provenance,
                    sentiment_summary=analysis['sentiment_summary'], topics=analysis['topics'],
                    skills=analysis['skills'], nlp_available=analysis['nlp_available'],
                    analysis_status=analysis.get('analysis_status', 'available' if analysis['nlp_available'] else 'unavailable'),
                    component_status=analysis.get('component_status', {}),
                    unavailable_reason=analysis.get('unavailable_reason'),
                )
            except (KeyError, TypeError, ValidationError) as exc:
                raise HTTPException(
                    status_code=503,
                    detail='NLP analysis failed governed schema validation; no result was cached.',
                ) from exc
            state.nlp_results = response.model_dump()
            return response
    except IntegrityError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
