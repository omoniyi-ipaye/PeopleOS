"""Fail closed when measured collaboration data and validated semantics are absent."""
from fastapi import APIRouter, Depends, HTTPException, Query
from api.dependencies import AppState, get_app_state

router = APIRouter(prefix='/api/network', tags=['network'])


def require_data(state: AppState = Depends(get_app_state)) -> AppState:
    if not state.has_data() and not state.load_from_database():
        raise HTTPException(status_code=400, detail='No data loaded. Please upload a file first.')
    return state


def unavailable():
    return HTTPException(status_code=409, detail=(
        'Collaboration analytics require observed relationship data and validated use-case semantics. '
        'Shared department and reporting lines do not establish collaboration, influence, isolation, or departure impact.'
    ))


@router.get('/influencers', deprecated=True)
async def get_key_influencers(limit: int = Query(default=10, ge=1, le=50), state: AppState = Depends(require_data)):
    raise unavailable()


@router.get('/isolated', deprecated=True)
async def get_isolated_employees(limit: int = Query(default=10, ge=1, le=50), state: AppState = Depends(require_data)):
    raise unavailable()


@router.get('/summary')
async def get_network_summary(state: AppState = Depends(require_data)):
    return {'success': False, 'available': False, 'reason': unavailable().detail}
