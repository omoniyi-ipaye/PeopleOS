"""Governed People Intelligence Agent API routes."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from api.dependencies import AppState, get_app_state
from src.agent.orchestrator import AgentAnswer, PeopleIntelligenceAgent
from src.platform.workspace import WorkspaceStore


router = APIRouter(prefix="/api/intelligence", tags=["people-intelligence"])
_store = WorkspaceStore()


class InvestigationRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    actor_id: Optional[str] = None
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
async def capabilities(state: AppState = Depends(get_app_state)) -> dict:
    agent = PeopleIntelligenceAgent(state)
    workspace = _store.ensure_workspace("local", "Local workspace")
    return {
        "agent": "People Intelligence Agent",
        "mode": "governed-read-only",
        "tools": agent.registry.list_ids(),
        "policy": agent.policy.policy_id,
        "llm_available": bool(
            getattr(state, "llm_client", None)
            and getattr(state.llm_client, "is_available", False)
        ),
        "data_loaded": state.has_data(),
        "workspace": workspace.model_dump(mode="json"),
    }


@router.post("/investigate", response_model=AgentAnswer)
async def investigate(
    request: InvestigationRequest,
    state: AppState = Depends(require_dataset),
) -> AgentAnswer:
    """Investigate a workforce question through governed aggregate tools."""
    workspace = _store.ensure_workspace(request.workspace_id)
    session = None
    if request.session_id:
        session = next((item for item in workspace.sessions if item.session_id == request.session_id), None)
        if session is None:
            raise HTTPException(status_code=404, detail="Unknown investigation session")
    else:
        session = _store.open_session(
            workspace_id=request.workspace_id,
            dataset_id=request.dataset_version or workspace.active_dataset_id,
            model_id=workspace.active_model_id,
        )

    dataset_id = request.dataset_version or session.dataset_id or workspace.active_dataset_id
    agent = PeopleIntelligenceAgent(state)
    answer = agent.investigate(
        request.question,
        actor_id=request.actor_id,
        workspace_id=request.workspace_id,
        dataset_version=dataset_id,
    )
    _store.record_request(
        request.workspace_id,
        session.session_id,
        answer.request_id,
        request.question,
    )
    return answer
