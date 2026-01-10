"""Semantic search route handlers."""

from typing import List, Dict, Any

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel

from api.dependencies import get_app_state, AppState

router = APIRouter(prefix="/api/search", tags=["search"])


class SearchResult(BaseModel):
    """Semantic search result."""
    employee_id: str
    dept: str
    text: str
    similarity_score: float


class SearchResponse(BaseModel):
    """Search response."""
    results: List[SearchResult]
    query: str
    total_results: int


def require_vector_search(state: AppState = Depends(get_app_state)) -> AppState:
    """Dependency that requires vector search engine."""
    if not state.has_data():
        if not state.load_from_database():
            raise HTTPException(
                status_code=400,
                detail="No data loaded. Please upload a file first."
            )

    if state.vector_engine is None or not state.vector_engine.is_initialized():
        raise HTTPException(
            status_code=400,
            detail="Semantic search requires PerformanceText column in data."
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
    results = state.vector_engine.search(query, top_k=top_k)

    search_results = []
    for result in results:
        search_results.append(SearchResult(
            employee_id=result['EmployeeID'],
            dept=result.get('Dept', 'Unknown'),
            text=result.get('text', ''),
            similarity_score=float(result['similarity_score'])
        ))

    return SearchResponse(
        results=search_results,
        query=query,
        total_results=len(search_results)
    )


@router.get("/status")
async def get_search_status(
    state: AppState = Depends(get_app_state)
) -> Dict[str, Any]:
    """
    Get status of the semantic search index.
    """
    if state.vector_engine is None:
        return {
            'available': False,
            'reason': 'Vector engine not initialized'
        }

    if not state.vector_engine.is_initialized():
        return {
            'available': False,
            'reason': 'Index not built (requires PerformanceText data)'
        }

    return {
        'available': True,
        'indexed_records': len(state.vector_engine.metadata),
        'embedding_dimension': state.vector_engine.dimension
    }
