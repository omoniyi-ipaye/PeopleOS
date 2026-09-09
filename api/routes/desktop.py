"""Desktop-only lifecycle endpoints for the packaged local application."""
from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, HTTPException

from desktop.control import controller

router = APIRouter(prefix="/api/desktop", tags=["desktop"])


def _require_desktop() -> None:
    if not controller.enabled:
        # Do not advertise process controls in normal web/source deployments.
        raise HTTPException(status_code=404, detail="Desktop controls are unavailable.")


@router.get("/status")
async def desktop_status():
    _require_desktop()
    return controller.status()


@router.post("/open")
async def desktop_open(background_tasks: BackgroundTasks):
    _require_desktop()
    background_tasks.add_task(controller.open)
    return {"accepted": True, "action": "open"}


@router.post("/restart")
async def desktop_restart(background_tasks: BackgroundTasks):
    _require_desktop()
    background_tasks.add_task(controller.request, "restart")
    return {"accepted": True, "action": "restart"}


@router.post("/quit")
async def desktop_quit(background_tasks: BackgroundTasks):
    _require_desktop()
    background_tasks.add_task(controller.request, "quit")
    return {"accepted": True, "action": "quit"}
