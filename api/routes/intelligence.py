"""Governed People Intelligence Agent API routes."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from api.dependencies import AppState, get_app_state
from src.agent.orchestrator import AgentAnswer, PeopleIntelligenceAgent


router = APIRouter(prefix="/api/intelligence", tags=["people-intelligence"])


class InvestigationRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    actor_id: Optional[str] = None
    workspace_id: Optional[str] = None
    dataset_version: Optional[str] = None


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
    }


@router.post("/investigate", response_model=AgentAnswer)
async def investigate(
    request: InvestigationRequest,
    state: AppState = Depends(require_dataset),
) -> AgentAnswer:
    """Investigate a workforce question through governed aggregate tools."""
    agent = PeopleIntelligenceAgent(state)
    return agent.investigate(
        request.question,
        actor_id=request.actor_id,
        workspace_id=request.workspace_id,
        dataset_version=request.dataset_version,
    )
