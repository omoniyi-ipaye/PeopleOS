"""Dataset identity and runtime evidence contracts shared by APIs and agents."""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd

from src.population import active_population
from src.data_contract import is_numeric_measurement


class IntegrityError(ValueError):
    """Evidence cannot be linked to a compatible, intact runtime snapshot."""


def frame_fingerprint(frame: pd.DataFrame) -> str:
    # Includes column names, values, row order and nulls; excludes incidental index.
    normalized = frame.reset_index(drop=True).copy()
    for column in normalized:
        if is_numeric_measurement(column) or column == 'Attrition':
            normalized[column] = pd.to_numeric(normalized[column], errors='coerce').astype(float)
        elif column == 'SnapshotDate':
            normalized[column] = pd.to_datetime(normalized[column], errors='coerce', utc=True, format='mixed')
        else:
            normalized[column] = normalized[column].astype('string')
    canonical = normalized.to_json(orient='split', date_format='iso', double_precision=15)
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def snapshot_provenance(state) -> dict[str, Any]:
    provenance = getattr(state, 'runtime_provenance', None)
    raw = getattr(state, 'raw_df', None)
    if not provenance or raw is None:
        raise IntegrityError('No verified dataset snapshot is active. Activate a dataset first.')
    if provenance.get('current_fingerprint') != frame_fingerprint(raw):
        raise IntegrityError('Runtime data changed after activation. Reactivate the dataset before analysis.')
    return dict(provenance)


def require_dataset_identity(state, workspace_id: str, dataset_id: str | None) -> dict:
    provenance = snapshot_provenance(state)
    if not dataset_id or provenance.get('workspace_id') != workspace_id or provenance.get('dataset_id') != dataset_id:
        raise IntegrityError('Requested dataset does not match the loaded snapshot. Activate that dataset before continuing.')
    return provenance


def validated_risk_scores(state) -> pd.DataFrame:
    """One authoritative probability/population/model boundary for all consumers."""
    provenance = snapshot_provenance(state)
    engine = getattr(state, 'ml_engine', None)
    if engine is None or not getattr(engine, 'is_trained', False) or getattr(state, 'model_metrics', None) is None:
        raise IntegrityError('No activated predictive model is available in the current runtime.')
    model_provenance = getattr(state, 'model_provenance', None) or {}
    if any(model_provenance.get(key) != provenance.get(key) for key in ('workspace_id', 'dataset_id', 'generation', 'current_fingerprint')):
        raise IntegrityError('Model output belongs to a different dataset snapshot. Activate a compatible model.')
    frame = getattr(state, 'risk_scores', None)
    if frame is None or frame.empty:
        raise IntegrityError('Aggregate predictive scores are not available.')
    frame = frame.copy()
    if not {'EmployeeID', 'risk_score'}.issubset(frame) or frame.EmployeeID.isna().any():
        raise IntegrityError('Predictive score identities are unavailable.')
    frame['EmployeeID'] = frame.EmployeeID.astype(str)
    scores = pd.to_numeric(frame.risk_score, errors='coerce')
    if frame.EmployeeID.duplicated().any() or not (np.isfinite(scores) & scores.between(0, 1)).all():
        raise IntegrityError('Predictive scores have duplicate identities or invalid probabilities.')
    current = active_population(state.raw_df)
    if 'EmployeeID' not in current or set(frame.EmployeeID) != set(current.EmployeeID.astype(str)):
        raise IntegrityError('Predictive scores do not exactly cover the current active population.')
    frame['risk_score'] = scores
    frame['risk_category'] = scores.map(engine.get_risk_category)
    if not frame.risk_category.isin(['High', 'Medium', 'Low']).all():
        raise IntegrityError('Predictive risk categories are unavailable.')
    return frame


def runtime_integrity(state, workspace=None) -> dict:
    """Observable integrity, distinct from statistical or enterprise validation."""
    issues = []
    provenance = getattr(state, 'runtime_provenance', None)
    try:
        provenance = snapshot_provenance(state)
        if workspace is not None:
            require_dataset_identity(state, workspace.workspace_id, workspace.active_dataset_id)
    except IntegrityError as exc:
        issues.append(str(exc))
    model_ready = False
    if getattr(state, 'ml_engine', None) is not None or (workspace is not None and workspace.active_model_id):
        try:
            validated_risk_scores(state)
            if workspace is not None and (getattr(state, 'model_provenance', None) or {}).get('model_id') != workspace.active_model_id:
                raise IntegrityError('Registered model and runtime model differ.')
            model_ready = True
        except IntegrityError as exc:
            issues.append(str(exc))
    return {'status': 'verified' if not issues else 'unavailable', 'issues': issues,
            'snapshot': provenance, 'model_ready': model_ready,
            'scope': 'dataset_and_runtime_integrity_not_statistical_validity'}
