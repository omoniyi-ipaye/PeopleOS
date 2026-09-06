"""Deterministic authorization policy for PeopleOS control-plane operations.

The API remains local-first. Loopback traffic is treated as the local owner.
Remote identity is established only after the bearer-token boundary succeeds;
its role is configured server-side with PEOPLEOS_API_ROLE and cannot be chosen
by a client request header.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet

from fastapi import HTTPException, Request


@dataclass(frozen=True)
class ActorContext:
    actor_id: str
    role: str
    local: bool


_PERMISSIONS: Dict[str, FrozenSet[str]] = {
    "owner": frozenset({"*"}),
    "admin": frozenset({
        "workspace.read", "workspace.write",
        "dataset.read", "dataset.write", "dataset.activate",
        "model.read", "model.train", "model.activate",
        "session.read", "session.write", "investigate",
        "health.read", "health.recover",
    }),
    "analyst": frozenset({
        "workspace.read", "dataset.read", "model.read",
        "session.read", "session.write", "investigate", "health.read",
    }),
    "viewer": frozenset({
        "workspace.read", "dataset.read", "model.read", "session.read", "health.read",
    }),
}


def actor_from_request(request: Request) -> ActorContext:
    role = getattr(request.state, "peopleos_role", "viewer")
    actor_id = getattr(request.state, "peopleos_actor_id", "anonymous")
    local = bool(getattr(request.state, "peopleos_local", False))
    return ActorContext(actor_id=actor_id, role=role, local=local)


def has_permission(role: str, permission: str) -> bool:
    allowed = _PERMISSIONS.get(role, frozenset())
    return "*" in allowed or permission in allowed


def require_permission(request: Request, permission: str) -> ActorContext:
    actor = actor_from_request(request)
    if not has_permission(actor.role, permission):
        raise HTTPException(
            status_code=403,
            detail=f"Role '{actor.role}' is not authorized for '{permission}'.",
        )
    return actor


def permissions_for_role(role: str) -> list[str]:
    return sorted(_PERMISSIONS.get(role, frozenset()))
