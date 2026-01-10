"""NLP analysis route handlers."""

from typing import List, Dict, Any, Optional

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from api.dependencies import get_app_state, AppState

router = APIRouter(prefix="/api/nlp", tags=["nlp"])


class SentimentSummary(BaseModel):
    """Sentiment analysis summary statistics."""
    avg_sentiment: float
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
    prevalence: str
    sentiment: Optional[str] = None
    count: Optional[int] = None


class NLPAnalysisResponse(BaseModel):
    """Full NLP analysis response."""
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
    if 'PerformanceText' not in state.raw_df.columns:
        raise HTTPException(
            status_code=400,
            detail="No PerformanceText column found in dataset"
        )

    # Return cached results if available
    if state.nlp_results is not None:
        return NLPAnalysisResponse(**state.nlp_results)

    analysis = state.nlp_engine.process_all(state.raw_df)
    
    # Cache results
    state.nlp_results = {
        'sentiment_summary': analysis['sentiment_summary'],
        'topics': analysis['topics'],
        'skills': analysis['skills'],
        'nlp_available': analysis['nlp_available']
    }

    return NLPAnalysisResponse(**state.nlp_results)
