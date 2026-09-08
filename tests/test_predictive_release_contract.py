"""Activation requires retained, matching evidence, not just a candidate label."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from src.platform.model_lifecycle import ModelEvaluationPolicy, ModelLifecycleService
from src.platform.workspace import ModelState, WorkspaceStore


@pytest.fixture
def release(tmp_path, monkeypatch):
    store = WorkspaceStore(str(tmp_path / 'workspace.json'))
    dataset = store.register_dataset(
        workspace_id='local', source_name='synthetic', content_hash='synthetic',
        row_count=200, columns=['EmployeeID', 'Salary', 'Attrition'],
        quality={'current_fingerprint': 'snapshot-A'},
    )
    store.activate_dataset('local', dataset.dataset_id)
    monkeypatch.setattr(ModelLifecycleService, '_runtime_artifacts', {})
    service = ModelLifecycleService(store)

    def candidate():
        metrics = dict(
            roc_auc=.8, brier_score=.12, baseline_brier_score=.25,
            average_precision=.8, baseline_average_precision=.5,
            calibration_error=.05, test_size=100, test_class_counts={'0': 50, '1': 50},
            cv_preprocessing_fold_local=True, holdout_untouched_by_fit=True,
            training_current_fingerprint='snapshot-A', future_departure_validated=False,
            evaluation_semantics='retrospective_employee_holdout_not_future_departure_validation',
        )
        model = store.create_model(workspace_id='local', dataset_id=dataset.dataset_id)
        store.update_model('local', model.model_id, state=ModelState.CANDIDATE,
                           metrics=metrics, evaluation=service.policy.evaluate(metrics))
        artifact = SimpleNamespace(metrics=deepcopy(metrics))
        service._runtime_artifacts[model.model_id] = artifact
        return model.model_id, artifact

    return store, service, candidate


def test_passing_release_retains_retrospective_scope(release):
    store, service, candidate = release
    model_id, _ = candidate()
    released = service.activate('local', model_id)
    assert released.state == ModelState.ACTIVE
    assert released.metrics['future_departure_validated'] is False
    assert released.evaluation['scope'] == 'minimum_retrospective_gate_not_enterprise_or_future_prediction_certification'
    assert store.get_workspace('local').active_model_id == model_id


@pytest.mark.parametrize('corruption', ['missing_gate', 'failed_gate', 'artifact_changed',
                                       'bad_metrics', 'snapshot_changed', 'missing_snapshot'])
def test_release_rejects_invalid_evidence_without_retiring_active_model(release, corruption):
    store, service, candidate = release
    previous, _ = candidate()
    service.activate('local', previous)
    model_id, artifact = candidate()
    if corruption in ('missing_gate', 'failed_gate'):
        store.update_model('local', model_id,
                           evaluation={} if corruption == 'missing_gate' else {'passed': False})
    elif corruption == 'artifact_changed':
        artifact.metrics['roc_auc'] = .9
    else:
        if corruption == 'bad_metrics':
            artifact.metrics['roc_auc'] = .5
        elif corruption == 'snapshot_changed':
            artifact.metrics['training_current_fingerprint'] = 'other-snapshot'
        else:
            artifact.metrics.pop('training_current_fingerprint')
        store.update_model('local', model_id, metrics=artifact.metrics)
    with pytest.raises(ValueError):
        service.activate('local', model_id)
    workspace = store.get_workspace('local')
    assert workspace.active_model_id == previous
    assert next(m for m in workspace.models if m.model_id == model_id).state == ModelState.CANDIDATE
    assert next(m for m in workspace.models if m.model_id == previous).state == ModelState.ACTIVE


def test_release_rechecks_current_policy(release):
    store, _, candidate = release
    model_id, _ = candidate()
    stricter = ModelLifecycleService(store, ModelEvaluationPolicy(min_auc=.85))
    with pytest.raises(ValueError, match='current retrospective evaluation gate'):
        stricter.activate('local', model_id)
    assert store.get_workspace('local').active_model_id is None
