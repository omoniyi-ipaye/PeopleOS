"""Durable canonical dataset artifacts for the local-first PeopleOS product.

SQLite remains a compatibility/current-row store, but every activated dataset
version is persisted independently so snapshot histories and future schema
extensions survive process restarts and application upgrades unchanged.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from src.local_paths import get_peopleos_paths


def dataset_artifact_path(dataset_id: str) -> Path:
    safe_id = ''.join(ch for ch in dataset_id if ch.isalnum() or ch in ('-', '_'))
    if not safe_id:
        raise ValueError('Invalid dataset id')
    return get_peopleos_paths().datasets / f'{safe_id}.csv'


def save_dataset_artifact(dataset_id: str, frame: pd.DataFrame) -> Path:
    """Atomically persist the full validated dataset version as CSV."""
    target = dataset_artifact_path(dataset_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix('.csv.tmp')
    frame.to_csv(tmp, index=False)
    os.replace(tmp, target)
    return target


def load_dataset_artifact(dataset_id: str) -> pd.DataFrame | None:
    target = dataset_artifact_path(dataset_id)
    if not target.exists():
        return None
    frame = pd.read_csv(target)
    return None if frame.empty else frame


def remove_dataset_artifact(dataset_id: str) -> None:
    target = dataset_artifact_path(dataset_id)
    if target.exists():
        target.unlink()
