"""Governed People Intelligence Agent API routes."""

from typing import Optional
from types import SimpleNamespace
from src.platform.runtime_lock import RUNTIME_MUTATION_LOCK

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from api.authorization import require_permission
from api.dependencies import AppState, get_app_state
from src.agent.orchestrator import AgentAnswer, PeopleIntelligenceAgent
from src.platform.workspace import WorkspaceStore
from src.platform.provenance import IntegrityError, require_dataset_identity


router = APIRouter(prefix="/api/intelligence", tags=["people-intelligence"])
_store = WorkspaceStore()


class InvestigationRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    workspace_id: str = "local"
    dataset_version: Optional[str] = None
    session_id: Optional[str] = None


def require_dataset(state: AppState = Depends(get_app_state)) -> AppState:
    if not state.has_data() and not state.load_from_database():
        raise HTTPException(
            status_code=400,
            detail="No workforce dataset is loaded. Upload data before starting an investigation.",
        )
    return state


@router.get("/capabilities")
async def capabilities(request: Request, state: AppState = Depends(get_app_state)) -> dict:
    require_permission(request, "workspace.read")
    agent = PeopleIntelligenceAgent(state)
    workspace = _store.ensure_workspace("local", "Local workspace")
    return {
        "agent": "People Intelligence Agent",
        "mode": "governed-read-only",
        "tools": agent.registry.list_ids(),
        "policy": agent.policy.policy_id,
        "llm_available": bool(getattr(state, "llm_client", None) and getattr(state.llm_client, "is_available", False)),
        "data_loaded": state.has_data(),
        "workspace": workspace.model_dump(mode="json"),
    }


@router.post("/investigate", response_model=AgentAnswer)
async def investigate(
    payload: InvestigationRequest,
    request: Request,
    state: AppState = Depends(require_dataset),
) -> AgentAnswer:
    """Investigate a workforce question through governed aggregate tools."""
    actor = require_permission(request, "investigate")
    workspace = _store.ensure_workspace(payload.workspace_id)

    try:
        require_dataset_identity(state, payload.workspace_id, payload.dataset_version or workspace.active_dataset_id)
    except IntegrityError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    if payload.session_id:
        session = next((item for item in workspace.sessions if item.session_id == payload.session_id), None)
        if session is None:
            raise HTTPException(status_code=404, detail="Unknown investigation session")
        if payload.dataset_version and payload.dataset_version != session.dataset_id:
            raise HTTPException(status_code=409, detail="An investigation session cannot change datasets. Start a new investigation.")
    else:
        session = _store.open_session(
            workspace_id=payload.workspace_id,
            dataset_id=payload.dataset_version or workspace.active_dataset_id,
            model_id=workspace.active_model_id,
        )

    dataset_id = payload.dataset_version or session.dataset_id or workspace.active_dataset_id
    try:
        require_dataset_identity(state, payload.workspace_id, dataset_id)
    except IntegrityError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    model_id = session.model_id or workspace.active_model_id
    if model_id and model_id != (getattr(state, 'model_provenance', None) or {}).get('model_id'):
        raise HTTPException(status_code=409, detail='Investigation model differs from the active runtime model. Start a new investigation.')
    with RUNTIME_MUTATION_LOCK:
        try:
            require_dataset_identity(state, payload.workspace_id, dataset_id)
        except IntegrityError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        analysis_state = SimpleNamespace(**state.__dict__)
    answer = PeopleIntelligenceAgent(analysis_state).investigate(
        payload.question,
        actor_id=actor.actor_id,
        workspace_id=payload.workspace_id,
        dataset_version=dataset_id,
        model_version=model_id,
    )
    with RUNTIME_MUTATION_LOCK:
        _store.record_request(payload.workspace_id, session.session_id, answer.request_id, payload.question)
    return answer
