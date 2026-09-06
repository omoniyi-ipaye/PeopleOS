"""Canonical JSON-safe serialization helpers for PeopleOS analytical outputs.

Analytics engines frequently produce NumPy/Pandas scalars and may surface NaN or
infinite values when a statistic is undefined. Public API contracts must never
leak those implementation details or emit invalid JSON. Undefined numeric values
are represented as ``None`` so the UI can present an explicit unavailable state.
"""

from __future__ import annotations

import math
from typing import Any


def json_safe(value: Any) -> Any:
    """Recursively convert analytical values into valid JSON-native values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]

    # NumPy scalar types and several Pandas scalar wrappers expose ``item``.
    item_method = getattr(value, 'item', None)
    if callable(item_method):
        try:
            native = item_method()
            if native is not value:
                return json_safe(native)
        except (TypeError, ValueError, OverflowError):
            pass

    isoformat = getattr(value, 'isoformat', None)
    if callable(isoformat):
        try:
            return isoformat()
        except (TypeError, ValueError, OverflowError):
            pass

    # Unknown analytical labels are preserved as strings instead of crashing an
    # otherwise useful response. Structured metrics should use explicit schemas.
    return str(value)
