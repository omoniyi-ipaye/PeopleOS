"""Regression tests for the local-first PeopleOS distribution contract."""

from pathlib import Path

from src.local_paths import get_peopleos_paths
from src.platform.workspace import WorkspaceStore
from src.utils import load_config


def test_peopleos_home_is_single_mutable_storage_root(monkeypatch, tmp_path: Path):
    home = tmp_path / "PeopleOS User Data"
    monkeypatch.setenv("PEOPLEOS_HOME", str(home))

    paths = get_peopleos_paths()
    assert paths.root == home.resolve()
    assert paths.database == home.resolve() / "data" / "peopleos.db"
    assert paths.registry == home.resolve() / "control-plane" / "workspaces.json"
    assert paths.datasets.parent == paths.root
    assert paths.models.parent == paths.root
    assert paths.backups.parent == paths.root
    assert all(path.exists() for path in (paths.root, paths.datasets, paths.models, paths.logs, paths.backups))


def test_config_database_path_follows_peopleos_home(monkeypatch, tmp_path: Path):
    home = tmp_path / "local-product"
    monkeypatch.setenv("PEOPLEOS_HOME", str(home))
    config = load_config()
    assert Path(config["persistence"]["database_path"]) == home.resolve() / "data" / "peopleos.db"


def test_workspace_registry_survives_process_state_outside_bundle(monkeypatch, tmp_path: Path):
    home = tmp_path / "durable"
    monkeypatch.setenv("PEOPLEOS_HOME", str(home))

    first = WorkspaceStore()
    dataset = first.register_dataset(
        workspace_id="local",
        source_name="workforce.csv",
        content_hash="abc123",
        row_count=42,
        columns=["EmployeeID", "Dept"],
    )
    first.activate_dataset("local", dataset.dataset_id)

    second = WorkspaceStore()
    workspace = second.get_workspace("local")
    assert workspace.active_dataset_id == dataset.dataset_id
    assert workspace.datasets[-1].source_name == "workforce.csv"
    assert second.path == home.resolve() / "control-plane" / "workspaces.json"
