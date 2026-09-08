"""Architecture invariants for the PeopleOS workspace control plane."""

import json

import pytest

from src.platform.health import SystemHealthMonitor
from src.platform.model_lifecycle import ModelEvaluationPolicy
from src.platform.workspace import DatasetState, ModelState, WorkspaceStore


def test_workspace_dataset_activation_versions_and_supersedes(tmp_path):
    store = WorkspaceStore(str(tmp_path / "workspaces.json"))
    first = store.register_dataset(
        workspace_id="acme",
        source_name="one.csv",
        content_hash="a" * 64,
        row_count=10,
        columns=["EmployeeID"],
    )
    second = store.register_dataset(
        workspace_id="acme",
        source_name="two.csv",
        content_hash="b" * 64,
        row_count=12,
        columns=["EmployeeID"],
    )
    assert first.version == 1
    assert second.version == 2
    store.activate_dataset("acme", first.dataset_id)
    store.activate_dataset("acme", second.dataset_id)
    workspace = store.get_workspace("acme")
    assert workspace.active_dataset_id == second.dataset_id
    assert next(d for d in workspace.datasets if d.dataset_id == first.dataset_id).state == DatasetState.SUPERSEDED
    assert next(d for d in workspace.datasets if d.dataset_id == second.dataset_id).state == DatasetState.ACTIVE


def test_model_activation_requires_evaluated_candidate(tmp_path):
    store = WorkspaceStore(str(tmp_path / "workspaces.json"))
    dataset = store.register_dataset(
        workspace_id="local",
        source_name="data.csv",
        content_hash="c" * 64,
        row_count=10,
        columns=["EmployeeID"],
    )
    store.activate_dataset("local", dataset.dataset_id)
    model = store.create_model(workspace_id="local", dataset_id=dataset.dataset_id)
    with pytest.raises(ValueError):
        store.activate_model("local", model.model_id)
    store.update_model("local", model.model_id, state=ModelState.CANDIDATE, evaluation={"passed": True})
    active = store.activate_model("local", model.model_id)
    assert active.state == ModelState.ACTIVE
    assert store.get_workspace("local").active_model_id == model.model_id


def test_evaluation_policy_requires_discrimination_calibration_and_leakage_safety():
    policy = ModelEvaluationPolicy(min_auc=0.60, max_brier=0.30)

    accepted = policy.evaluate({
        "baseline_brier_score": 0.25, "average_precision": 0.7, "baseline_average_precision": 0.5,
        "calibration_error": 0.05, "cv_preprocessing_fold_local": True,
        "test_size": 100, "test_class_counts": {"0": 50, "1": 50},
        "roc_auc": 0.72,
        "brier_score": 0.20,
        "holdout_untouched_by_fit": True,
    })
    assert accepted["passed"] is True

    assert policy.evaluate({
        "roc_auc": 0.51,
        "brier_score": 0.20,
        "holdout_untouched_by_fit": True,
    })["passed"] is False
    assert policy.evaluate({
        "roc_auc": 0.72,
        "brier_score": 0.40,
        "holdout_untouched_by_fit": True,
    })["passed"] is False
    assert policy.evaluate({
        "roc_auc": 0.72,
        "brier_score": 0.20,
        "holdout_untouched_by_fit": False,
    })["passed"] is False
    assert policy.evaluate({"roc_auc": 0.72})["passed"] is False
    assert policy.evaluate({})["passed"] is False


def test_session_persists_hash_not_question(tmp_path):
    store = WorkspaceStore(str(tmp_path / "workspaces.json"))
    session = store.open_session(workspace_id="local")
    question = "Why is turnover changing?"
    store.record_request("local", session.session_id, "pia_123", question)
    raw = (tmp_path / "workspaces.json").read_text(encoding="utf-8")
    assert question not in raw
    persisted = store.get_workspace("local").sessions[0]
    assert persisted.request_ids == ["pia_123"]
    assert persisted.question_hashes == [store.hash_text(question)]


def test_health_recovery_is_bounded_to_control_plane(tmp_path):
    path = tmp_path / "broken.json"
    store = WorkspaceStore(str(path))
    original = b"not-json"
    path.write_bytes(original)
    with pytest.raises(RuntimeError, match="original bytes were retained"):
        WorkspaceStore(str(path))
    assert path.read_bytes() == original
    result = SystemHealthMonitor(store).recover()
    assert (tmp_path / result["quarantine"]["filename"]).read_bytes() == original
    assert result["status"] == "metadata_reinitialized"
    assert result["requires_review"] is True
    health = result["health"]
    assert health["status"] == "degraded"
    assert "activate a model" in health["governed_only"]
    assert "change employee data" in health["governed_only"]
    assert store.get_workspace("local").active_dataset_id is None
    assert store.get_workspace("local").active_model_id is None


def test_registry_file_is_valid_json_after_atomic_writes(tmp_path):
    path = tmp_path / "workspaces.json"
    store = WorkspaceStore(str(path))
    store.ensure_workspace("beta", "Beta")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert any(item["workspace_id"] == "beta" for item in payload["workspaces"])
