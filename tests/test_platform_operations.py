"""Operational invariants for authorization, monitoring and jobs."""

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from api.authorization import has_permission, require_permission
from src.platform.jobs import JobState, JobStore
from src.platform.monitoring import FitnessPolicy
from src.platform.workspace import ModelState, WorkspaceStore


def _request(role: str):
    return SimpleNamespace(state=SimpleNamespace(
        peopleos_role=role,
        peopleos_actor_id=f"{role}-actor",
        peopleos_local=role == "owner",
    ))


def test_authorization_policy_blocks_consequential_platform_controls():
    assert has_permission("analyst", "investigate") is True
    assert has_permission("analyst", "model.activate") is False
    assert has_permission("viewer", "model.train") is False
    assert has_permission("admin", "model.activate") is True
    assert has_permission("owner", "anything") is True
    with pytest.raises(HTTPException) as exc:
        require_permission(_request("analyst"), "health.recover")
    assert exc.value.status_code == 403


def test_job_idempotency_and_interruption_recovery(tmp_path):
    jobs = JobStore(str(tmp_path / "jobs.json"))
    first = jobs.create(workspace_id="local", kind="model-training", idempotency_key="same")
    second = jobs.create(workspace_id="local", kind="model-training", idempotency_key="same")
    assert first.job_id == second.job_id
    jobs.transition(first.job_id, JobState.RUNNING)
    recovered = jobs.recover_interrupted()
    assert recovered == [first.job_id]
    assert jobs.list()[0].state == JobState.FAILED
    assert "safe retry" in (jobs.list()[0].error or "")


def test_fitness_policy_detects_missing_active_dataset(tmp_path):
    store = WorkspaceStore(str(tmp_path / "workspaces.json"))
    workspace = store.get_workspace("local")
    fitness = FitnessPolicy().assess(workspace)
    assert fitness["status"] == "degraded"
    assert fitness["checks"]["active_dataset_present"] is False


def test_fitness_policy_accepts_fresh_dataset_and_quality_model(tmp_path):
    store = WorkspaceStore(str(tmp_path / "workspaces.json"))
    dataset = store.register_dataset(
        workspace_id="local", source_name="data.csv", content_hash="d" * 64,
        row_count=100, columns=["EmployeeID"]
    )
    store.activate_dataset("local", dataset.dataset_id)
    model = store.create_model(workspace_id="local", dataset_id=dataset.dataset_id)
    store.update_model(
        "local", model.model_id, state=ModelState.CANDIDATE,
        metrics={"roc_auc": 0.75}, evaluation={"passed": True}
    )
    store.activate_model("local", model.model_id)
    fitness = FitnessPolicy().assess(store.get_workspace("local"))
    assert fitness["status"] == "healthy"
    assert fitness["observed"]["model_auc"] == 0.75
