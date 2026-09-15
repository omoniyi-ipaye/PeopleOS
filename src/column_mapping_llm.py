"""Optional local-AI assistance for workforce column mapping.

The mapper is deliberately narrower than PeopleOS's HR advisor. It receives
column names and shape/type metadata only; employee cell values never leave the
upload route and are not included in the prompt. Suggestions are advisory and
must be reviewed by the user before an explicit upload mapping is committed.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

import pandas as pd

from src.data_loader import GOLDEN_SCHEMA
from src.llm_client import LLMClient, LLMClientError
from src.logger import get_logger


logger = get_logger("column_mapping_llm")

_DATA_SCOPE = "Column names and shape metadata only; workforce cell values are not sent to local AI."


def _profile_columns(frame: pd.DataFrame) -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = []
    for column in frame.columns:
        source = str(column)
        series = frame[column]
        text = series.astype("string")
        nonempty = text[text.str.strip().ne("") & text.notna()]
        numeric_ratio = float(pd.to_numeric(nonempty, errors="coerce").notna().mean()) if len(nonempty) else 0.0
        date_ratio = float(pd.to_datetime(nonempty, format="mixed", errors="coerce", utc=True).notna().mean()) if len(nonempty) else 0.0
        inferred = str(pd.api.types.infer_dtype(nonempty, skipna=True)) if len(nonempty) else "empty"
        profiles.append({
            "source": source,
            "inferred_type": inferred,
            "row_count": int(len(series)),
            "nonempty_count": int(len(nonempty)),
            "unique_count": int(nonempty.nunique(dropna=True)),
            "numeric_ratio": round(numeric_ratio, 3),
            "date_ratio": round(date_ratio, 3),
        })
    return profiles


def _parse_json_response(raw: Any) -> list[dict[str, Any]]:
    text = raw if isinstance(raw, str) else ""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        # Some local models add a short preamble or trailing explanation even
        # in JSON mode. Parse only the first complete object and still reject
        # malformed or truncated output rather than guessing a mapping.
        start = text.find("{")
        if start < 0:
            raise
        decoder = json.JSONDecoder()
        payload, _ = decoder.raw_decode(text[start:])
    if isinstance(payload, dict):
        payload = payload.get("mappings", [])
    if not isinstance(payload, list):
        raise ValueError("Local AI did not return a mapping list")
    return [item for item in payload if isinstance(item, dict)]


def suggest_column_mappings(
    frame: pd.DataFrame,
    *,
    existing_mapping: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Return validated local-LLM suggestions for unresolved source columns."""
    try:
        client = LLMClient(respect_preferences=True)
    except Exception as exc:
        logger.warning("Local mapping AI could not be initialized: %s", type(exc).__name__)
        return {"available": False, "used": False, "mappings": {}, "details": [], "reason": "Local AI is not ready."}

    if not client.is_available:
        return {
            "available": False,
            "used": False,
            "mappings": {},
            "details": [],
            "reason": client.unavailable_reason or "Local AI is not ready.",
        }

    source_columns = [str(column) for column in frame.columns]
    existing = {str(source): str(target) for source, target in (existing_mapping or {}).items() if target}
    allowed_fields = GOLDEN_SCHEMA["required"] + GOLDEN_SCHEMA["optional"]
    unresolved = [source for source in source_columns if source not in existing]
    if not unresolved:
        return {
            "available": True,
            "used": False,
            "mappings": {},
            "details": [],
            "reason": "PeopleOS already mapped every source column it can use.",
        }

    profiles = [profile for profile in _profile_columns(frame) if profile["source"] in unresolved]
    prompt = (
        "You are a careful data-schema assistant for PeopleOS. Map source workforce export columns "
        "to the allowed PeopleOS canonical fields. This is schema mapping only, not HR advice.\n\n"
        "Return JSON only in this shape: {\"mappings\":[{\"source\":\"source header\","
        "\"target\":\"AllowedField\",\"confidence\":0.0,\"reason\":\"short reason\"}]}\n"
        "Only use an exact allowed target or null when there is no safe match. Do not invent fields. "
        "Do not map one target from more than one source. Critical fields still require human review.\n\n"
        f"Allowed fields: {json.dumps(allowed_fields)}\n"
        f"Unresolved source column metadata: {json.dumps(profiles, separators=(',', ':'))}\n"
    )

    try:
        raw = client.generate(
            prompt,
            format="json",
            options={"temperature": 0, "num_predict": 1200},
        )
        items = _parse_json_response(raw)
    except (LLMClientError, ValueError, TypeError, json.JSONDecodeError) as exc:
        logger.warning("Local mapping AI returned no usable suggestions: %s", type(exc).__name__)
        return {
            "available": True,
            "used": False,
            "mappings": {},
            "details": [],
            "reason": "Local AI returned no usable mapping suggestions. Review the deterministic mapping instead.",
        }

    allowed = set(allowed_fields)
    source_set = set(unresolved)
    used_targets = set(existing.values())
    mappings: dict[str, str] = {}
    details: list[dict[str, Any]] = []
    for item in items:
        source = item.get("source")
        target = item.get("target")
        if not isinstance(source, str) or source not in source_set:
            continue
        if target is None:
            continue
        if not isinstance(target, str) or target not in allowed or target in used_targets:
            continue
        try:
            confidence = max(0.0, min(1.0, float(item.get("confidence", 0.6))))
        except (TypeError, ValueError):
            confidence = 0.6
        mappings[source] = target
        used_targets.add(target)
        details.append({
            "source": source,
            "target": target,
            "method": "llm",
            "confidence": confidence,
            "reason": str(item.get("reason") or "Suggested from the local schema assistant."),
        })

    return {
        "available": True,
        "used": bool(mappings),
        "mappings": mappings,
        "details": details,
        "reason": None if mappings else "Local AI found no additional safe mappings.",
    }


def mapping_data_scope() -> str:
    return _DATA_SCOPE
