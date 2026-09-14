"""Semantic search route handlers."""

import math
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel

from api.dependencies import get_app_state, AppState
from src.platform.provenance import IntegrityError, snapshot_provenance

router = APIRouter(prefix="/api/search", tags=["search"])


class SearchResult(BaseModel):
    """Semantic search result."""
    dept: str
    text: str
    similarity_score: float
    squared_l2_distance: float | None = None
    score_semantics: str = 'inverse_squared_l2_distance_not_probability_or_validated_relevance'


class SearchProvenance(BaseModel):
    """Safe dataset binding for retrieved evidence, without worker identifiers."""
    workspace_id: Optional[str] = None
    dataset_id: Optional[str] = None
    generation: Optional[str] = None
    current_fingerprint: Optional[str] = None


class SearchResponse(BaseModel):
    """Search response."""
    results: List[SearchResult]
    query: str
    total_results: int
    provenance: SearchProvenance


def require_vector_search(state: AppState = Depends(get_app_state)) -> AppState:
    """Dependency that requires vector search engine."""
    if not state.has_data():
        if not state.load_from_database():
            raise HTTPException(
                status_code=400,
                detail="No data loaded. Please upload a file first."
            )

    raw = getattr(state, 'raw_df', None)
    if raw is None or 'PerformanceText' not in raw.columns or not raw['PerformanceText'].notna().any():
        raise HTTPException(
            status_code=400,
            detail="Semantic search requires PerformanceText column in data."
        )

    engine = getattr(state, 'vector_engine', None)
    if engine is None or not engine.is_initialized():
        raise HTTPException(
            status_code=400,
            detail="Semantic search is unavailable; the optional embedding backend or index is not ready."
        )
    try:
        provenance = snapshot_provenance(state)
    except IntegrityError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    matches_provenance = getattr(engine, 'matches_provenance', None)
    if not callable(matches_provenance) or not matches_provenance(provenance):
        raise HTTPException(
            status_code=409,
            detail="Semantic search is unavailable; the index does not match the active dataset snapshot."
        )

    return state


@router.post("", response_model=SearchResponse)
async def search_performance_reviews(
    query: str = Query(..., min_length=3, description="Search query"),
    top_k: int = Query(default=10, ge=1, le=50, description="Number of results"),
    state: AppState = Depends(require_vector_search)
) -> SearchResponse:
    """
    Search performance reviews using semantic similarity.

    The search uses sentence-transformers embeddings and FAISS
    for efficient similarity search.
    """
    try:
        provenance = snapshot_provenance(state)
    except IntegrityError as exc:
        # The supported dependency has already validated provenance. Keep a
        # narrow compatibility path for older injected test/legacy adapters;
        # a real VectorEngine always has matches_provenance and fails closed.
        if callable(getattr(state.vector_engine, 'matches_provenance', None)):
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        provenance = None
    try:
        results = state.vector_engine.search(query, top_k=top_k, provenance=provenance)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=503, detail="Semantic search is unavailable; no relevance result was produced.") from exc

    search_results = []
    for result in results:
        try:
            score = float(result['similarity_score'])
            distance = result.get('squared_l2_distance')
            if distance is not None:
                distance = float(distance)
            if not all(value is None or math.isfinite(value) for value in (score, distance)):
                raise ValueError('Non-finite ranking evidence')
            search_results.append(SearchResult(
                dept=str(result.get('Dept', 'Unknown')),
                text=str(result.get('text', result.get('PerformanceText', ''))),
                similarity_score=score,
                squared_l2_distance=distance,
            ))
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=503, detail="Semantic search is unavailable; malformed ranking evidence was discarded.") from exc

    try:
        current_provenance = snapshot_provenance(state)
    except IntegrityError as exc:
        if provenance is not None:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        current_provenance = None
    if provenance is not None and current_provenance != provenance:
        raise HTTPException(status_code=409, detail='Semantic search is unavailable; the active dataset changed during retrieval.')

    return SearchResponse(
        results=search_results,
        query=query,
        total_results=len(search_results),
        provenance=SearchProvenance(**{key: provenance.get(key) for key in SearchProvenance.model_fields}) if provenance else SearchProvenance(),
    )


@router.get("/status")
async def get_search_status(
    state: AppState = Depends(get_app_state)
) -> Dict[str, Any]:
    """
    Get status of the semantic search index.
    """
    engine = getattr(state, 'vector_engine', None)
    if engine is None:
        return {
            'available': False,
            'reason': 'Optional embedding backend is not initialized; semantic search is unavailable.'
        }

    if not engine.is_initialized():
        return {
            'available': False,
            'reason': 'Semantic search index is not built for the active dataset.'
        }

    try:
        current = snapshot_provenance(state)
    except IntegrityError:
        return {
            'available': False,
            'reason': 'Semantic search is unavailable; no verified dataset snapshot is active.'
        }
    matches_provenance = getattr(engine, 'matches_provenance', None)
    if not callable(matches_provenance) or not matches_provenance(current):
        return {
            'available': False,
            'reason': 'Semantic search is unavailable; the index does not match the active dataset snapshot.'
        }

    provenance = getattr(engine, 'index_provenance', None) or {}
    return {
        'available': True,
        'indexed_records': len(engine.metadata),
        'embedding_dimension': engine.dimension,
        'index_dataset_id': provenance.get('dataset_id'),
    }
