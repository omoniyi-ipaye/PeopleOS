"""Cross-process probe for the PeopleOS local restart contract.

Usage:
  python tests/local_restart_probe.py load
  python tests/local_restart_probe.py restore

Both invocations must use the same PEOPLEOS_HOME. The second process proves the
workspace survives a real process boundary rather than only an in-memory reset.
"""

from __future__ import annotations

import sys
from pathlib import Path

from api.dependencies import AppState


SAMPLE = Path(__file__).resolve().parents[1] / "sample_hr_data.csv"


def load() -> None:
    state = AppState()
    state.reset()
    result = state.load_data(str(SAMPLE), SAMPLE.name)
    assert result["rows_loaded"] >= 50
    assert state.has_data()
    print(f"LOCAL RESTART LOAD: PASS ({result['rows_loaded']} rows)")


def restore() -> None:
    state = AppState()
    assert state.load_from_database(), "No persisted local workforce was restored"
    assert state.has_data()
    assert state.raw_df is not None and len(state.raw_df) >= 50
    assert state.analytics_engine is not None
    print(f"LOCAL RESTART RESTORE: PASS ({len(state.raw_df)} rows)")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "load":
        load()
    elif mode == "restore":
        restore()
    else:
        raise SystemExit("Expected load or restore")
