"""Governed People Intelligence Agent API routes."""

import asyncio
from typing import Optional
from types import SimpleNamespace
from src.platform.runtime_lock import RUNTIME_MUTATION_LOCK

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from api.authorization import require_permission
from api.dependencies import AppState, get_app_state
from src.agent.orchestrator import AgentAnswer
from src.agent.governed_agent import GovernedPeopleIntelligenceAgent
from src.agent.tools import AgentToolDescriptor
from src.platform.workspace import WorkspaceStore
from src.platform.provenance import IntegrityError, require_dataset_identity


router = APIRouter(prefix="/api/intelligence", tags=["people-intelligence"])
_store = WorkspaceStore()


class InvestigationRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    workspace_id: str = "local"
    dataset_version: Optional[str] = None
    session_id: Optional[str] = None
    agentic: bool = True


def require_dataset(state: AppState = Depends(get_app_state)) -> AppState:
    if not state.has_data() and not state.load_from_database():
        raise HTTPException(status_code=400, detail="No workforce dataset is loaded. Upload data before starting an investigation.")
    return state


@router.get("/capabilities")
async def capabilities(request: Request, state: AppState = Depends(get_app_state)) -> dict:
    require_permission(request, "workspace.read")
    agent = GovernedPeopleIntelligenceAgent(state)
    workspace = _store.ensure_workspace("local", "Local workspace")
    tool_catalog = agent.registry.list_descriptors()
    tool_catalog.append(AgentToolDescriptor(
        tool_id=agent.derived_tool.tool_id,
        description=agent.derived_tool.description,
        engine='analysis_sandbox',
        api_routes=('/api/intelligence/investigate',),
        availability='available',
    ).as_dict())
    return {
        "agent": "People Intelligence Agent",
        "mode": "governed-read-only",
        "tools": [item['tool_id'] for item in tool_catalog],
        "tool_catalog": tool_catalog,
        # The catalog is the source of truth for the read surfaces. Keeping
        # this list aligned with the executable registry prevents a capability
        # response from claiming less (or more) access than the agent has.
        "api_read_surfaces": tool_catalog,
        "agentic_tool_selection": {
            "enabled": True,
            "execution_order": "deterministic_plan_then_local_model_selection_then_grounded_explanation",
            "max_additional_tools": 4,
            "model_can_select": "allowlisted read-only tool IDs only",
        },
        "data_access": {
            "full_active_snapshot_internal_read": True,
            "model_context": "schema_and_redacted_aggregate_evidence",
            "raw_records_to_model": False,
            "employee_identifiers_to_model": False,
            "free_text_to_model": False,
            "reason": "PeopleOS scans the complete snapshot in-process for governed calculations, then redacts row-level and free-text material before local-model interpretation.",
        },
        "policy": agent.policy.policy_id,
        "llm_available": bool(getattr(state, "llm_client", None) and getattr(state.llm_client, "is_available", False)),
        "derived_analysis": {
            "enabled": True,
            "execution": "typed-deterministic",
            "row_level_output": False,
            "shell_access": False,
            "network_access": False,
        },
        "data_loaded": state.has_data(),
        "workspace": workspace.model_dump(mode="json"),
    }


@router.post("/investigate", response_model=AgentAnswer)
async def investigate(payload: InvestigationRequest, request: Request, state: AppState = Depends(require_dataset)) -> AgentAnswer:
    """Investigate a workforce question through governed aggregate tools."""
    return await _run_investigation(payload, request, state, persist_session=True, record_audit=True)


async def _run_investigation(
    payload: InvestigationRequest,
    request: Request,
    state: AppState,
    *,
    persist_session: bool,
    record_audit: bool,
) -> AgentAnswer:
    """Run one investigation with an explicit durable-side-effect policy."""
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
        if session.actor_id and session.actor_id != actor.actor_id:
            raise HTTPException(status_code=403, detail="This investigation session belongs to a different actor.")
        if payload.dataset_version and payload.dataset_version != session.dataset_id:
            raise HTTPException(status_code=409, detail="An investigation session cannot change datasets. Start a new investigation.")
    else:
        if persist_session:
            session = _store.open_session(
                workspace_id=payload.workspace_id,
                dataset_id=payload.dataset_version or workspace.active_dataset_id,
                model_id=workspace.active_model_id,
                actor_id=actor.actor_id,
            )
        else:
            session = SimpleNamespace(
                session_id=None,
                dataset_id=payload.dataset_version or workspace.active_dataset_id or
                (getattr(state, 'runtime_provenance', None) or {}).get('dataset_id'),
                model_id=workspace.active_model_id,
            )

    dataset_id = payload.dataset_version or session.dataset_id or workspace.active_dataset_id or (getattr(state, 'runtime_provenance', None) or {}).get('dataset_id')
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

    # Evidence planning is synchronous and the optional local synthesis path
    # can wait on Ollama. Keep that work off FastAPI's event loop so a slow
    # model cannot freeze health, navigation or lock requests.
    answer = await asyncio.to_thread(
        GovernedPeopleIntelligenceAgent(analysis_state).investigate,
        payload.question,
        actor_id=actor.actor_id,
        workspace_id=payload.workspace_id,
        dataset_version=dataset_id,
        model_version=model_id,
        record_audit=record_audit,
        agentic=payload.agentic,
    )
    if persist_session:
        with RUNTIME_MUTATION_LOCK:
            _store.record_request(payload.workspace_id, session.session_id, answer.request_id, payload.question)
    return answer
