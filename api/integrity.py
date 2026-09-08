"""Contain results from requests that overlap a dataset/model switch."""
from fastapi import HTTPException
from fastapi.responses import JSONResponse

from api.runtime_registry import get_workspace_state
from src.platform.provenance import IntegrityError, snapshot_provenance
from src.platform.runtime_lock import RUNTIME_MUTATION_LOCK

_EVIDENCE_PREFIXES = tuple('/api/' + name for name in (
    'analytics', 'predictions', 'compensation', 'succession', 'team', 'fairness',
    'search', 'advisor', 'nlp', 'survival', 'quality-of-hire', 'structural',
    'sentiment', 'experience', 'scenario', 'model-lab', 'geo', 'causal', 'network', 'intelligence',
))


async def evidence_snapshot_guard(request, call_next):
    if not request.url.path.startswith(_EVIDENCE_PREFIXES):
        return await call_next(request)
    try:
        state = get_workspace_state(request)
        with RUNTIME_MUTATION_LOCK:
            before = snapshot_provenance(state) if state.has_data() else None
            model_before = (getattr(state, 'model_provenance', None) or {}).get('model_id')
    except (IntegrityError, HTTPException) as exc:
        return JSONResponse(status_code=getattr(exc, 'status_code', 409), content={'detail': getattr(exc, 'detail', str(exc))})
    response = await call_next(request)
    if response.status_code >= 400:
        return response
    try:
        with RUNTIME_MUTATION_LOCK:
            if not state.has_data():
                if before is not None:
                    raise IntegrityError('Dataset was reset during analysis. Load a dataset and retry.')
                return response
            after = snapshot_provenance(state)
            model_after = (getattr(state, 'model_provenance', None) or {}).get('model_id')
    except IntegrityError as exc:
        return JSONResponse(status_code=409, content={'detail': str(exc)})
    if before is not None and (before != after or model_before != model_after):
        return JSONResponse(status_code=409, content={'detail': 'Dataset or model changed during analysis. Retry using the current snapshot.'})
    response.headers['X-PeopleOS-Snapshot'] = after['generation']
    if after.get('dataset_id'):
        response.headers['X-PeopleOS-Dataset'] = after['dataset_id']
    return response
