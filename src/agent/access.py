"""Privacy-preserving data access helpers for the PeopleOS agent.

The deterministic runtime may inspect the complete active snapshot in-process
to calculate a result. This module defines what can cross the agent/model
boundary: schema and coverage are safe, while identifiers, free text and
employee-level records are redacted before a result is placed in an evidence
bundle or model prompt.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from src.serialization import json_safe


_SENSITIVE_FIELD = re.compile(
    r'(^|_)(employee_?id|manager_?id|person_?id|first_?name|last_?name|full_?name|'
    r'name|email|phone|mobile|address|street|ssn|social_?security|passport|'
    r'national_?id|nie|dni|performance_?text|review_?text|free_?text|comment|'
    r'feedback|raw_?text|survey_?response)(_|$)',
    re.IGNORECASE,
)
_ROW_IDENTIFIER = re.compile(
    r'^(?:employee_?id|manager_?id|person_?id|first_?name|last_?name|full_?name|'
    r'name|email|phone|mobile|address|street|ssn|social_?security|passport|'
    r'national_?id|nie|dni)$',
    re.IGNORECASE,
)
_SENSITIVE_COLLECTION = re.compile(
    r'^(?:employee_?ids?|employee_?rows?|employees?|people|persons?|at_?risk_?employees?|'
    r'high_?potentials?|risk_?scores?|names?|texts?|comments?|feedback|reviews?)$',
    re.IGNORECASE,
)
_MAX_LIST_ITEMS = 120
_MAX_DEPTH = 8
_REDACTED = object()

# Model prompts need a smaller, decision-useful view than the complete
# evidence ledger. The server still retains the full redacted engine result;
# these limits only apply to the optional narrative model context.
_MODEL_CONTEXT_MAX_DEPTH = 5
_MODEL_CONTEXT_FIT_STEPS = (
    (5, 12, 8, 240),
    (4, 10, 6, 200),
    (3, 8, 4, 160),
    (2, 6, 3, 120),
)


def _normalized_key(value: Any) -> str:
    return re.sub(r'[^a-z0-9]+', '_', str(value).strip().lower()).strip('_')


def is_sensitive_field(name: Any) -> bool:
    """Return whether a field can identify a person or carry free text."""
    normalized = _normalized_key(name)
    return bool(_SENSITIVE_FIELD.search(normalized))


def _is_row_identifier(name: Any) -> bool:
    return bool(_ROW_IDENTIFIER.fullmatch(_normalized_key(name)))


def _is_sensitive_collection(name: Any) -> bool:
    return bool(_SENSITIVE_COLLECTION.fullmatch(_normalized_key(name)))


def redact_for_agent(value: Any) -> Tuple[Any, int]:
    """Recursively remove employee-level and free-text material.

    The returned count is operational metadata only; it lets the UI explain
    why an engine result is smaller without persisting the removed values.
    """

    redactions = 0

    def walk(current: Any, *, key: Any = None, depth: int = 0) -> Any:
        nonlocal redactions
        if depth > _MAX_DEPTH:
            redactions += 1
            return '[truncated for agent context]'
        if key is not None and (is_sensitive_field(key) or _is_sensitive_collection(key)):
            redactions += 1
            return _REDACTED
        if isinstance(current, pd.DataFrame):
            columns = list(current.columns)
            if any(is_sensitive_field(column) for column in columns):
                redactions += 1
                return _REDACTED
            return walk(current.to_dict(orient='records'), depth=depth + 1)
        if isinstance(current, pd.Series):
            return walk(current.to_dict(), depth=depth + 1)
        if isinstance(current, dict):
            # A row-like object with an identifier is removed as a whole. It is
            # not safe to keep its remaining score/measurement fields anonymous.
            if any(_is_row_identifier(item_key) for item_key in current):
                redactions += 1
                return _REDACTED
            result: Dict[str, Any] = {}
            for item_key, item_value in current.items():
                cleaned = walk(item_value, key=item_key, depth=depth + 1)
                if cleaned is not _REDACTED:
                    result[str(item_key)] = cleaned
            return result
        if isinstance(current, (list, tuple, set)):
            result = []
            for item in list(current)[:_MAX_LIST_ITEMS]:
                cleaned = walk(item, depth=depth + 1)
                if cleaned is not _REDACTED:
                    result.append(cleaned)
            if len(current) > _MAX_LIST_ITEMS:
                redactions += 1
                result.append({'omitted_items': len(current) - _MAX_LIST_ITEMS})
            return result
        if isinstance(current, (np.generic,)):
            current = current.item()
        try:
            return json_safe(current)
        except Exception:
            redactions += 1
            return '[unavailable for agent context]'

    cleaned = walk(value)
    return (None if cleaned is _REDACTED else cleaned), redactions


def compact_for_agent_context(value: Any, *, max_chars: int = 1800) -> Tuple[Any, int]:
    """Return a bounded, redacted representation for the narrative model.

    Analytical engines may legitimately return curves, cohort tables or other
    large structures for the UI and evidence ledger. Sending those structures
    verbatim to a local language model is slow and gives the model more detail
    than it needs to explain the headline finding. This helper preserves
    scalar leaves, summary fields and collection shape while bounding depth,
    list size and serialized size. It never broadens the privacy boundary:
    redaction runs before compaction, and the complete redacted result remains
    available to the server-side response.
    """

    cleaned, redactions = redact_for_agent(value)
    if cleaned is None:
        return None, redactions

    def render(candidate: Any) -> str:
        return json.dumps(candidate, default=str, separators=(',', ':'))

    def compact(current: Any, *, depth: int, dict_limit: int, list_limit: int, string_limit: int) -> Any:
        if isinstance(current, str):
            if len(current) <= string_limit:
                return current
            return current[: max(0, string_limit - 28)] + '… [text truncated]'
        if isinstance(current, dict):
            keys = list(current)
            if depth >= _MODEL_CONTEXT_MAX_DEPTH:
                scalar_fields = {
                    str(key): item
                    for key, item in current.items()
                    if isinstance(item, (str, int, float, bool)) or item is None
                }
                result = {
                    key: compact(item, depth=depth + 1, dict_limit=dict_limit, list_limit=list_limit, string_limit=string_limit)
                    for key, item in list(scalar_fields.items())[:dict_limit]
                }
                result['_available_fields'] = [str(key) for key in keys[:dict_limit]]
                if len(keys) > dict_limit:
                    result['_omitted_fields'] = len(keys) - dict_limit
                return result
            result = {}
            for key in keys[:dict_limit]:
                result[str(key)] = compact(
                    current[key],
                    depth=depth + 1,
                    dict_limit=dict_limit,
                    list_limit=list_limit,
                    string_limit=string_limit,
                )
            if len(keys) > dict_limit:
                result['_omitted_fields'] = len(keys) - dict_limit
            return result
        if isinstance(current, (list, tuple, set)):
            values = list(current)
            if len(values) <= list_limit:
                kept = values
            else:
                head = max(1, list_limit // 2)
                tail = max(0, list_limit - head)
                kept = values[:head] + (values[-tail:] if tail else [])
            result = [
                compact(
                    item,
                    depth=depth + 1,
                    dict_limit=dict_limit,
                    list_limit=list_limit,
                    string_limit=string_limit,
                )
                for item in kept
            ]
            if len(values) > len(kept):
                result.append({'_omitted_items': len(values) - len(kept)})
            return result
        if isinstance(current, (np.generic,)):
            return current.item()
        return current

    for depth, dict_limit, list_limit, string_limit in _MODEL_CONTEXT_FIT_STEPS:
        candidate = compact(
            cleaned,
            depth=0,
            dict_limit=dict_limit,
            list_limit=list_limit,
            string_limit=string_limit,
        )
        if len(render(candidate)) <= max_chars:
            return candidate, redactions

    # The fallback contains structure only. It is intentionally derived from
    # keys, not values, so an unexpectedly shaped engine result cannot bypass
    # the context-size or row-level privacy boundary.
    if isinstance(cleaned, dict):
        fallback: Any = {
            'context_truncated': True,
            'available_fields': [str(key) for key in list(cleaned)[:24]],
            'field_count': len(cleaned),
        }
    elif isinstance(cleaned, (list, tuple, set)):
        fallback = {'context_truncated': True, 'item_count': len(cleaned)}
    else:
        fallback = {'context_truncated': True, 'value_type': type(cleaned).__name__}
    return fallback, redactions + 1


def _field_type(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return 'boolean'
    if pd.api.types.is_numeric_dtype(series):
        return 'numeric'
    if pd.api.types.is_datetime64_any_dtype(series):
        return 'date'
    return 'text_or_category'


def profile_frame(frame: pd.DataFrame) -> Dict[str, Any]:
    """Scan the complete frame and return schema/coverage, never sample values."""
    columns = []
    for name in frame.columns:
        series = frame[name]
        non_null = int(series.notna().sum())
        columns.append({
            'name': str(name),
            'field_type': _field_type(series),
            'dtype': str(series.dtype),
            'row_count': int(len(series)),
            'non_null_count': non_null,
            'missing_count': int(len(series) - non_null),
            'coverage_pct': round(non_null / len(series) * 100, 2) if len(series) else None,
            'distinct_count': int(series.nunique(dropna=True)),
            'protected': is_sensitive_field(name),
        })
    protected = [item['name'] for item in columns if item['protected']]
    return {
        'records_scanned': int(len(frame)),
        'columns': columns,
        'column_count': len(columns),
        'protected_field_count': len(protected),
        'protected_fields': protected,
        'full_scan': True,
        'raw_values_included': False,
        'employee_rows_included': False,
        'access_semantics': 'Complete active snapshot scanned in-process; only schema, counts and aggregate-safe results may cross the model boundary.',
    }
