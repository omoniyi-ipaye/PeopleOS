"""Validate required invariants in model/system.json."""

from __future__ import annotations

import json
from pathlib import Path


def validate(path: str = "model/system.json") -> list[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    errors: list[str] = []
    required = ["system_id", "purpose", "stage", "boundary", "components", "flows", "controls", "health", "build_steps"]
    for key in required:
        if key not in payload:
            errors.append(f"missing required key: {key}")
    if payload.get("stage") not in {"AS-IS", "TRANSITION", "TARGET"}:
        errors.append("stage must be AS-IS, TRANSITION or TARGET")
    component_ids = {item.get("id") for item in payload.get("components", [])}
    actor_ids = {item.get("id") for item in payload.get("actors", [])}
    valid_nodes = component_ids | actor_ids
    for flow in payload.get("flows", []):
        if flow.get("from") not in valid_nodes:
            errors.append(f"flow {flow.get('id')} has unknown source")
        if flow.get("to") not in valid_nodes:
            errors.append(f"flow {flow.get('id')} has unknown target")
    build_ids = [item.get("id") for item in payload.get("build_steps", [])]
    if len(build_ids) != len(set(build_ids)):
        errors.append("build step IDs must be unique")
    if not any(control.get("enforced") for control in payload.get("controls", [])):
        errors.append("at least one enforceable control is required")
    health = payload.get("health", {})
    if health.get("autonomy_level") not in {"L0", "L1", "L2", "L3"}:
        errors.append("health.autonomy_level must be L0-L3")
    if not health.get("maximum_blast_radius"):
        errors.append("health.maximum_blast_radius is required")
    return errors


if __name__ == "__main__":
    failures = validate()
    if failures:
        for failure in failures:
            print(f"ERROR: {failure}")
        raise SystemExit(1)
    print("PeopleOS canonical system model is valid")
