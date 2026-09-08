"""Workspace, dataset/model lifecycle, session, monitoring and health API."""

from __future__ import annotations

from typing import Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from api.authorization import actor_from_request, permissions_for_role, require_permission
from api.dependencies import AppState, get_app_state
from src.platform.health import SystemHealthMonitor
from src.platform.local_dataset_store import load_dataset_artifact, save_dataset_artifact, remove_dataset_artifact, dataset_artifact_path
from src.platform.runtime_loader import prepare_dataframe
from src.platform.provenance import IntegrityError, require_dataset_identity, runtime_integrity, validated_risk_scores
from types import SimpleNamespace
from uuid import uuid4
from src.platform.runtime_lock import RUNTIME_MUTATION_LOCK
from src.platform.jobs import JobState, JobStore
from src.platform.model_lifecycle import ModelLifecycleService
from src.platform.monitoring import FitnessPolicy
from src.platform.workspace import WorkspaceStore

router = APIRouter(prefix='/api/platform', tags=['platform'])
_store = WorkspaceStore()
_jobs = JobStore()


class WorkspaceRequest(BaseModel):
    workspace_id: str = Field(min_length=1, max_length=80, pattern=r'^[A-Za-z0-9_-]+$')
    name: str = Field(min_length=1, max_length=120)


class SessionRequest(BaseModel):
    workspace_id: str = 'local'
    dataset_id: Optional[str] = None
    model_id: Optional[str] = None


@router.get('/me')
async def who_am_i(request: Request):
    actor = actor_from_request(request)
    return {**actor.__dict__, 'permissions': permissions_for_role(actor.role)}


@router.get('/workspaces')
async def list_workspaces(request: Request):
    require_permission(request, 'workspace.read')
    return [item.model_dump(mode='json') for item in _store.list_workspaces()]


@router.post('/workspaces')
async def create_workspace(payload: WorkspaceRequest, request: Request):
    require_permission(request, 'workspace.write')
    return _store.ensure_workspace(payload.workspace_id, payload.name).model_dump(mode='json')


@router.get('/workspaces/{workspace_id}')
async def get_workspace(workspace_id: str, request: Request):
    require_permission(request, 'workspace.read')
    try:
        return _store.get_workspace(workspace_id).model_dump(mode='json')
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get('/workspaces/{workspace_id}/fitness')
async def workspace_fitness(workspace_id: str, request: Request):
    require_permission(request, 'health.read')
    try:
        return FitnessPolicy().assess(_store.get_workspace(workspace_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post('/workspaces/{workspace_id}/datasets/current')
async def register_current_dataset(workspace_id: str, request: Request, state: AppState = Depends(get_app_state)):
    with RUNTIME_MUTATION_LOCK:
        require_permission(request, 'dataset.write')
        if workspace_id != getattr(state, 'workspace_id', 'local'):
            raise HTTPException(status_code=409, detail='Requested workspace has no compatible data runtime.')
        if not state.has_data():
            raise HTTPException(status_code=400, detail='No dataset is currently loaded')
        csv_bytes = state.raw_df.to_csv(index=False).encode('utf-8')
        source = state.historical_df if state.historical_df is not None else state.raw_df
        dataset_id = f'ds_{uuid4().hex}'
        path = save_dataset_artifact(dataset_id, source)
        try:
            record = _store.register_dataset(
                workspace_id=workspace_id,
                dataset_id=dataset_id,
                source_name='current-loaded-dataset',
                content_hash=_store.hash_bytes(csv_bytes),
                row_count=len(state.raw_df),
                columns=list(state.raw_df.columns),
                quality={
                    'missing_cells': int(state.raw_df.isna().sum().sum()),
                    'duplicate_rows': int(state.raw_df.duplicated().sum()),
                    'artifact_sha256': _store.hash_bytes(path.read_bytes()),
                    'current_fingerprint': state.runtime_provenance['current_fingerprint'],
                },
            )
            return record.model_dump(mode='json')
        except Exception:
            remove_dataset_artifact(dataset_id)
            raise


@router.post('/workspaces/{workspace_id}/datasets/{dataset_id}/activate')
async def activate_dataset(workspace_id: str, dataset_id: str, request: Request, state: AppState = Depends(get_app_state)):
    with RUNTIME_MUTATION_LOCK:
        require_permission(request, 'dataset.activate')
        if workspace_id != getattr(state, 'workspace_id', 'local'):
            raise HTTPException(status_code=409, detail='Requested workspace has no compatible data runtime.')
        try:
            workspace = _store.get_workspace(workspace_id)
            record = next((d for d in workspace.datasets if d.dataset_id == dataset_id), None)
            if record is None:
                raise KeyError('Unknown dataset')
            source = load_dataset_artifact(dataset_id, expected_sha256=record.quality.get('artifact_sha256'))
            if source is None:
                raise ValueError('Dataset artifact is unavailable. Import it again before activation.')
            candidate, _ = prepare_dataframe(state, source, workspace_id=workspace_id, dataset_id=dataset_id)
            candidate.runtime_provenance.update(dataset_version=record.version, source_name=record.source_name)
            result = _store.activate_dataset(workspace_id, dataset_id)
            state.__dict__.update(candidate.__dict__)
            return result.model_dump(mode='json')
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post('/workspaces/{workspace_id}/models/train')
async def train_model(workspace_id: str, request: Request, state: AppState = Depends(get_app_state)):
    with RUNTIME_MUTATION_LOCK:
        actor = require_permission(request, 'model.train')
        workspace = _store.get_workspace(workspace_id)
        if not workspace.active_dataset_id:
            raise HTTPException(status_code=409, detail='Activate a dataset before training')
        try:
            require_dataset_identity(state, workspace_id, workspace.active_dataset_id)
        except IntegrityError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if state.raw_df is None or 'Attrition' not in state.raw_df.columns or not state.features_enabled.get('predictive', False):
            raise HTTPException(status_code=400, detail='Current dataset does not support safe predictive model training')

        idempotency_key = request.headers.get('idempotency-key') or f'train:{workspace_id}:{workspace.active_dataset_id}:{actor.actor_id}'
        job = _jobs.create(workspace_id=workspace_id, kind='model-training', idempotency_key=idempotency_key)
        if job.state == JobState.SUCCEEDED:
            artifact = ModelLifecycleService.runtime_artifact(job.resource_id)
            current_hash = state.runtime_provenance['current_fingerprint']
            prior_model = next((m for m in workspace.models if m.model_id == job.resource_id), None)
            reusable = prior_model is not None and prior_model.state.value in {'candidate', 'active', 'rejected'}
            if reusable and artifact is not None and getattr(artifact.engine, 'is_trained', False) and artifact.metrics.get('training_current_fingerprint') == current_hash:
                return {'job': job.model_dump(mode='json'), 'model_id': job.resource_id, 'reused': True}
            # Metadata survives restart; in-memory models do not. A real retry must fit again.
            job = _jobs.create(workspace_id=workspace_id, kind='model-training', idempotency_key=f'{idempotency_key}:recovery:{uuid4().hex}')
        _jobs.transition(job.job_id, JobState.RUNNING)
        try:
            model = ModelLifecycleService(_store).train(workspace_id, workspace.active_dataset_id, state.raw_df.copy())
            completed = _jobs.transition(job.job_id, JobState.SUCCEEDED, resource_id=model.model_id)
            return {'job': completed.model_dump(mode='json'), 'model': model.model_dump(mode='json'), 'reused': False}
        except Exception as exc:
            _jobs.transition(job.job_id, JobState.FAILED, error=type(exc).__name__)
            raise HTTPException(status_code=422, detail='Model training failed validation or evaluation checks') from exc


@router.post('/workspaces/{workspace_id}/models/{model_id}/activate')
async def activate_model(workspace_id: str, model_id: str, request: Request, state: AppState = Depends(get_app_state)):
    with RUNTIME_MUTATION_LOCK:
        require_permission(request, 'model.activate')
        service = ModelLifecycleService(_store)
        try:
            workspace = _store.get_workspace(workspace_id)
            provenance = require_dataset_identity(state, workspace_id, workspace.active_dataset_id)
            model = next((m for m in workspace.models if m.model_id == model_id), None)
            if model is None:
                raise KeyError('Unknown model')
            if model.dataset_id != provenance['dataset_id']:
                raise ValueError('Model dataset differs from the active snapshot')
            artifact = service.runtime_artifact(model_id)
            if artifact is None:
                raise ValueError('Runtime model artifact unavailable')
            if artifact.metrics.get('training_current_fingerprint') != provenance['current_fingerprint']:
                raise ValueError('Model training data does not match the active snapshot')
            engine = artifact.engine
            scoring_population = getattr(state, 'active_df', None)
            if scoring_population is None:
                scoring_population = state.raw_df[state.raw_df['Attrition'] == 0].copy() if state.raw_df is not None and 'Attrition' in state.raw_df.columns else state.raw_df
            if scoring_population is None or scoring_population.empty:
                raise ValueError('No active population is available for predictive scoring')

            processed = engine.preprocessor.transform(scoring_population, target_column='Attrition')
            X = processed.reindex(columns=engine.feature_names, fill_value=0)
            scores = engine.predict_risk(X)

            candidate = SimpleNamespace(**state.__dict__)
            candidate.model_provenance = {**provenance, 'model_id': model_id}
            candidate.ml_engine = engine
            candidate.model_metrics = artifact.metrics
            candidate.features_df = X.reset_index(drop=True)
            candidate.target_series = None
            # Do not manufacture employee-level probability confidence intervals from
            # tree disagreement or fixed +/- bands. Uncertainty is represented by the
            # model's held-out calibration/Brier metrics at aggregate level.
            candidate.risk_scores = pd.DataFrame({
                'EmployeeID': scoring_population['EmployeeID'].astype(str).values,
                'risk_score': scores,
                'risk_category': [engine.get_risk_category(score) for score in scores],
            })

            try:
                from src.fairness_engine import FairnessEngine
                prediction_frame = candidate.risk_scores[['EmployeeID', 'risk_score']].copy()
                prediction_frame['predicted'] = (prediction_frame['risk_score'] >= engine.risk_threshold_medium).astype(int)
                candidate.fairness_engine = FairnessEngine(candidate.raw_df, prediction_frame)
            except Exception:
                candidate.fairness_engine = None
            validated_risk_scores(candidate)
            model = service.activate(workspace_id, model_id)
            state.__dict__.update(candidate.__dict__)
            return model.model_dump(mode='json')
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post('/sessions')
async def open_session(payload: SessionRequest, request: Request):
    actor = require_permission(request, 'session.write')
    try:
        return _store.open_session(workspace_id=payload.workspace_id, dataset_id=payload.dataset_id, model_id=payload.model_id, actor_id=actor.actor_id).model_dump(mode='json')
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get('/jobs')
async def list_jobs(request: Request, workspace_id: Optional[str] = None):
    require_permission(request, 'health.read')
    return [job.model_dump(mode='json') for job in _jobs.list(workspace_id)]


@router.get('/health')
async def platform_health(request: Request, state: AppState = Depends(get_app_state)):
    require_permission(request, 'health.read')
    health = SystemHealthMonitor(_store).check()
    workspace_id = (getattr(state, 'runtime_provenance', None) or {}).get('workspace_id', 'local')
    health['runtime_integrity'] = runtime_integrity(state, _store.get_workspace(workspace_id))
    if state.has_data() and health['runtime_integrity']['status'] != 'verified':
        health['status'] = 'degraded'
    health['interrupted_jobs'] = [job.job_id for job in _jobs.list() if job.state == JobState.RUNNING]
    return health


@router.post('/health/recover')
async def bounded_recovery(request: Request):
    require_permission(request, 'health.recover')
    from api.runtime_registry import runtime_registry, WorkspaceRuntimeState
    with RUNTIME_MUTATION_LOCK:
        # Prepare first: a configuration failure must not follow metadata repair.
        replacement = WorkspaceRuntimeState('local')
        interrupted = _jobs.recover_interrupted()
        result = SystemHealthMonitor(_store).recover()
        if result['status'] == 'metadata_reinitialized':
            result['invalidated_runtimes'] = runtime_registry.invalidate_loaded(replacement)
            ModelLifecycleService._runtime_artifacts.clear()
            result['actions'].append('retired orphaned runtime evidence and in-memory model artifacts; source files retained')
        result['failed_interrupted_jobs'] = interrupted
        return result
