"""Owner-controlled local AI setup and readiness endpoints."""

from __future__ import annotations

from typing import Any, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from api.authorization import require_permission
from api.dependencies import AppState, get_app_state
from src.platform.ai_runtime import (
    AIPreferencesStore,
    LocalLLMSetupError,
    local_llm_setup,
    local_llm_status,
    refresh_llm_state,
    test_ollama_model,
)
from src.platform.runtime_lock import RUNTIME_MUTATION_LOCK


router = APIRouter(prefix="/api/llm", tags=["llm"])


class InstalledModel(BaseModel):
    name: str
    digest: Optional[str] = None
    size: Optional[int] = None
    remote: bool = False


class LLMStatus(BaseModel):
    provider: Literal["none", "ollama"]
    enabled: bool
    ready: bool
    ollama_installed: bool
    ollama_running: bool
    ollama_binary: Optional[str] = None
    host: str
    selected_model: str
    selected_model_installed: bool
    selected_model_digest: Optional[str] = None
    installed_models: list[InstalledModel] = Field(default_factory=list)
    recommended_model: str
    reason: Optional[str] = None
    download_guide: str
    setup_state: str
    setup_progress: int
    setup_message: Optional[str] = None
    setup_error_code: Optional[str] = None
    setup_model: Optional[str] = None


class LLMConfigureRequest(BaseModel):
    provider: Literal["none", "ollama"] = "ollama"
    enabled: bool
    model: Optional[str] = Field(default=None, min_length=1, max_length=128)


class LLMSetupRequest(BaseModel):
    model: Optional[str] = Field(default=None, min_length=1, max_length=128)


def _require_local_owner(request: Request) -> None:
    if not getattr(request.state, "peopleos_local", False) or getattr(request.state, "peopleos_role", None) != "owner":
        raise HTTPException(status_code=403, detail="Only the local PeopleOS owner can manage local AI.")


@router.get("/status", response_model=LLMStatus)
async def get_llm_status(request: Request) -> LLMStatus:
    require_permission(request, "sensitive.read")
    return LLMStatus.model_validate(local_llm_status())


@router.post("/configure", response_model=LLMStatus)
async def configure_llm(
    payload: LLMConfigureRequest,
    request: Request,
    state: AppState = Depends(get_app_state),
) -> LLMStatus:
    _require_local_owner(request)
    with RUNTIME_MUTATION_LOCK:
        try:
            AIPreferencesStore().update(
                enabled=payload.enabled,
                model=payload.model,
                provider=payload.provider,
            )
            refresh_llm_state(state)
        except (LocalLLMSetupError, RuntimeError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
    return LLMStatus.model_validate(local_llm_status())


@router.post("/setup", response_model=LLMStatus)
async def setup_llm(payload: LLMSetupRequest, request: Request) -> LLMStatus:
    _require_local_owner(request)
    current = local_llm_status()
    selected = payload.model or current["recommended_model"]
    try:
        local_llm_setup.start(selected)
    except (LocalLLMSetupError, RuntimeError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return LLMStatus.model_validate(local_llm_status())


@router.post("/test")
async def test_llm(request: Request) -> dict[str, Any]:
    _require_local_owner(request)
    current = local_llm_status()
    if not current["ready"]:
        raise HTTPException(status_code=409, detail=current["reason"] or "Local AI is not ready.")
    try:
        result = test_ollama_model(current["host"], current["selected_model"])
    except LocalLLMSetupError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"passed": bool(result["passed"]), **result}
