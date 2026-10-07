"""Semantic search route handlers."""

import importlib.util
import math
from typing import List, Dict, Any, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Depends, Query, Request
from pydantic import BaseModel

from api.authorization import require_permission
from api.dependencies import get_app_state, AppState
from src.logger import get_logger
from src.platform.provenance import IntegrityError, snapshot_provenance
from src.platform.runtime_lock import RUNTIME_MUTATION_LOCK

router = APIRouter(prefix="/api/search", tags=["search"])
logger = get_logger('search_route')


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


class SearchStatus(BaseModel):
    """Operational state of optional semantic retrieval."""
    available: bool
    state: str
    backend_available: bool
    can_prepare: bool
    reason: Optional[str] = None
    indexed_records: int = 0
    embedding_dimension: Optional[int] = None
    index_dataset_id: Optional[str] = None


def _embedding_backend_installed() -> bool:
    """Check the optional modules without importing or downloading a model."""
    try:
        return all(importlib.util.find_spec(name) is not None
                   for name in ('faiss', 'sentence_transformers'))
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _text_records(state: AppState) -> tuple[list[str], list[dict[str, Any]]]:
    """Return only nonempty current-snapshot text and non-identifying metadata."""
    raw = getattr(state, 'raw_df', None)
    if raw is None or 'PerformanceText' not in raw.columns:
        return [], []

    texts = raw['PerformanceText'].fillna('').astype(str).str.strip()
    departments = (
        raw['Dept'].fillna('Unknown').astype(str).str.strip()
        if 'Dept' in raw.columns else pd.Series('Unknown', index=raw.index)
    )
    valid = texts.ne('')
    selected_texts = texts[valid].tolist()
    selected_departments = departments[valid].replace('', 'Unknown').tolist()
    metadata = [
        {'Dept': department, 'text': text}
        for department, text in zip(selected_departments, selected_texts)
    ]
    return selected_texts, metadata


def _load_state_if_needed(state: AppState) -> None:
    """Restore the active local artifact for status/prepare after a restart."""
    if not state.has_data():
        state.load_from_database()


def _status_payload(state: AppState) -> dict[str, Any]:
    """Build a truthful, actionable status without initializing heavy dependencies."""
    _load_state_if_needed(state)
    backend_available = _embedding_backend_installed()
    texts, _ = _text_records(state)

    if not state.has_data():
        return SearchStatus(
            available=False,
            state='no_dataset',
            backend_available=backend_available,
            can_prepare=False,
            reason='Add or activate a workforce dataset before preparing semantic search.',
        ).model_dump(mode='json')
    if not texts:
        return SearchStatus(
            available=False,
            state='no_text',
            backend_available=backend_available,
            can_prepare=False,
            reason='The active dataset has no nonempty PerformanceText evidence to index.',
        ).model_dump(mode='json')
    engine = getattr(state, 'vector_engine', None)
    initialized = bool(engine is not None and engine.is_initialized())
    # A successfully prepared in-memory index remains usable for the life of
    # the process even when package discovery is unavailable later (for
    # example, a frozen desktop runtime). Check the backend before offering a
    # new preparation, not before reporting an already valid index.
    if not initialized and not backend_available:
        return SearchStatus(
            available=False,
            state='backend_unavailable',
            backend_available=False,
            can_prepare=False,
            reason='Install requirements-advanced.txt to enable local semantic search. The structured evidence path remains available.',
        ).model_dump(mode='json')
    if not initialized:
        return SearchStatus(
            available=False,
            state='not_prepared',
            backend_available=True,
            can_prepare=True,
            reason='The optional embedding backend is ready. Prepare an index for the active dataset to enable semantic search.',
        ).model_dump(mode='json')

    try:
        current = snapshot_provenance(state)
    except IntegrityError:
        return SearchStatus(
            available=False,
            state='unverified_snapshot',
            backend_available=True,
            can_prepare=False,
            reason='The active dataset snapshot is not verified. Reactivate the dataset before preparing search.',
        ).model_dump(mode='json')
    matches_provenance = getattr(engine, 'matches_provenance', None)
    if not callable(matches_provenance) or not matches_provenance(current):
        return SearchStatus(
            available=False,
            state='stale_index',
            backend_available=True,
            can_prepare=True,
            reason='The existing index belongs to a different dataset snapshot. Prepare it again for the active dataset.',
        ).model_dump(mode='json')

    provenance = getattr(engine, 'index_provenance', None) or {}
    return SearchStatus(
        available=True,
        state='ready',
        backend_available=True,
        can_prepare=True,
        indexed_records=len(getattr(engine, 'metadata', [])),
        embedding_dimension=getattr(engine, 'dimension', None),
        index_dataset_id=provenance.get('dataset_id'),
    ).model_dump(mode='json')


def _new_vector_engine():
    """Lazy-load the optional engine only after the user explicitly prepares it."""
    from src.vector_engine import VectorEngine
    return VectorEngine()


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


@router.post('/prepare', response_model=SearchStatus)
async def prepare_search_index(
    request: Request,
    state: AppState = Depends(get_app_state),
) -> SearchStatus:
    """Prepare a local, in-memory index bound to the active dataset snapshot."""
    require_permission(request, 'sensitive.read')
    with RUNTIME_MUTATION_LOCK:
        _load_state_if_needed(state)
        if not state.has_data():
            raise HTTPException(status_code=400, detail='Add or activate a workforce dataset before preparing semantic search.')

        texts, metadata = _text_records(state)
        if not texts:
            raise HTTPException(status_code=400, detail='The active dataset has no nonempty PerformanceText evidence to index.')
        try:
            provenance = snapshot_provenance(state)
        except IntegrityError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        try:
            engine = _new_vector_engine()
            engine.build_index(texts, metadata, provenance=provenance)
        except Exception as exc:
            logger.warning('Semantic search preparation failed: %s', exc)
            raise HTTPException(
                status_code=503,
                detail='Local semantic search could not prepare. Install requirements-advanced.txt and verify the pinned embedding model can load locally.',
            ) from exc

        # The new engine is assigned only after a complete, provenance-bound
        # replacement index exists. A failed refresh cannot expose partial data.
        state.vector_engine = engine
        return SearchStatus.model_validate(_status_payload(state))


@router.get('/status', response_model=SearchStatus)
async def get_search_status(
    state: AppState = Depends(get_app_state)
) -> SearchStatus:
    """
    Get status of the semantic search index.
    """
    with RUNTIME_MUTATION_LOCK:
        return SearchStatus.model_validate(_status_payload(state))
