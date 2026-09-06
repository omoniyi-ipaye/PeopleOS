"""Cross-process probe for the PeopleOS local restart contract.

Usage:
  python tests/local_restart_probe.py load
  python tests/local_restart_probe.py restore

Both invocations use the same PEOPLEOS_HOME. The probe intentionally loads a
snapshot-history dataset, which bypasses the legacy one-row-per-employee SQLite
store. Therefore a successful second process proves canonical dataset-artifact
restoration rather than an in-memory or SQLite fallback.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from api.dependencies import AppState
from api.routes.upload import _register_loaded_dataset
from src.platform.workspace import WorkspaceStore


SAMPLE = ROOT / "sample_hr_data.csv"


def _snapshot_file() -> Path:
    home = Path(os.environ["PEOPLEOS_HOME"])
    home.mkdir(parents=True, exist_ok=True)
    target = home / "restart-snapshot-history.csv"
    base = pd.read_csv(SAMPLE)
    older = base.copy(); older["SnapshotDate"] = "2026-01-31"
    newer = base.copy(); newer["SnapshotDate"] = "2026-06-30"
    pd.concat([older, newer], ignore_index=True).to_csv(target, index=False)
    return target


def load() -> None:
    state = AppState()
    state.reset()
    source = _snapshot_file()
    content = source.read_bytes()
    result = state.load_data(str(source), source.name)
    assert result["rows_loaded"] >= 50
    assert result["snapshot_history"] is True
    assert result["source_rows"] == result["rows_loaded"] * 2
    dataset = _register_loaded_dataset(state, source.name, WorkspaceStore.hash_bytes(content))
    assert dataset.dataset_id == WorkspaceStore().get_workspace("local").active_dataset_id
    assert state.has_data()
    print(f"LOCAL RESTART LOAD: PASS ({result['source_rows']} historical rows)")


def restore() -> None:
    state = AppState()
    assert state.load_from_database(), "No persisted local workforce was restored"
    assert state.has_data()
    assert state.raw_df is not None and len(state.raw_df) >= 50
    assert state.historical_df is not None
    assert len(state.historical_df) == len(state.raw_df) * 2
    assert state.population_resolution is not None and state.population_resolution.snapshot_history is True
    assert state.analytics_engine is not None
    print(f"LOCAL RESTART RESTORE: PASS ({len(state.historical_df)} historical rows)")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "load":
        load()
    elif mode == "restore":
        restore()
    else:
        raise SystemExit("Expected load or restore")
