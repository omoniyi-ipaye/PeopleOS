"""NLP analysis route handlers."""

from typing import List, Dict, Any, Optional

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

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


class TopicInfo(BaseModel):
    """NLP topic information."""
    name: str
    description: str
    prevalence: Optional[str] = None
    measurement_semantics: str = "generated_theme_not_measured_prevalence"
    sample_size: Optional[int] = None
    sentiment: Optional[str] = None
    count: Optional[int] = None


class NLPAnalysisResponse(BaseModel):
    """Full NLP analysis response."""
    provenance: Optional[Dict[str, Any]] = None
    sentiment_summary: SentimentSummary
    topics: List[TopicInfo]
    skills: Dict[str, Any]
    nlp_available: bool


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
                return NLPAnalysisResponse(**state.nlp_results)
            engine = state.nlp_engine
            if engine is None:
                raise HTTPException(status_code=503, detail='NLP engine not initialized')
            source = state.raw_df.copy(deep=True)
    except IntegrityError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    # Inference may take minutes; it must not block dataset/reset mutations.
    analysis = engine.process_all(source)
    try:
        with RUNTIME_MUTATION_LOCK:
            if snapshot_provenance(state) != provenance:
                raise IntegrityError('Dataset changed while text analysis was running. Retry the analysis.')
            response = NLPAnalysisResponse(
                provenance=provenance,
                sentiment_summary=analysis['sentiment_summary'], topics=analysis['topics'],
                skills=analysis['skills'], nlp_available=analysis['nlp_available'],
            )
            state.nlp_results = response.model_dump()
            return response
    except IntegrityError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
