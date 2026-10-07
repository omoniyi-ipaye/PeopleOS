"""Local owner app-lock endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from src.platform.app_lock import AppLockError, AppLockStore


router = APIRouter(prefix="/api/app-lock", tags=["app-lock"])


class PinRequest(BaseModel):
    pin: str


class PinChangeRequest(BaseModel):
    current_pin: str
    new_pin: str


def _require_local_owner(request: Request) -> None:
    if not getattr(request.state, "peopleos_local", False) or getattr(request.state, "peopleos_role", None) != "owner":
        raise HTTPException(status_code=403, detail="Only the local PeopleOS owner can manage the app lock.")


@router.get("/status")
async def app_lock_status():
    return AppLockStore().status()


@router.post("/setup")
async def app_lock_setup(request: Request, payload: PinRequest):
    _require_local_owner(request)
    try:
        return AppLockStore().setup(payload.pin)
    except AppLockError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/lock")
async def app_lock_lock(request: Request):
    _require_local_owner(request)
    try:
        return AppLockStore().lock()
    except AppLockError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/unlock")
async def app_lock_unlock(request: Request, payload: PinRequest):
    _require_local_owner(request)
    try:
        return AppLockStore().unlock(payload.pin)
    except AppLockError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


@router.post("/change")
async def app_lock_change(request: Request, payload: PinChangeRequest):
    _require_local_owner(request)
    try:
        return AppLockStore().change_pin(payload.current_pin, payload.new_pin)
    except AppLockError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


@router.post("/disable")
async def app_lock_disable(request: Request, payload: PinRequest):
    _require_local_owner(request)
    try:
        return AppLockStore().disable(payload.pin)
    except AppLockError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
