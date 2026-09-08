"""Regression tests for the local-first PeopleOS distribution contract."""

from pathlib import Path

import pandas as pd
import pytest

from src import utils
from src.local_paths import get_peopleos_paths
from src.platform.local_dataset_store import load_dataset_artifact, save_dataset_artifact
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


@pytest.mark.parametrize("source_directory_exists", [False, True])
@pytest.mark.parametrize("use_peopleos_home", [False, True])
def test_config_loads_from_bundle_without_requiring_source_directory(
    monkeypatch, tmp_path: Path, source_directory_exists: bool, use_peopleos_home: bool
):
    bundle = tmp_path / "bundle with spaces"
    bundle.mkdir()
    if source_directory_exists:
        (bundle / "src").mkdir()
    config_text = "version: bundle-fixture\npersistence:\n  database_path: data/peopleos.db\n"
    (bundle / "config.yaml").write_text(config_text, encoding="utf-8")
    # PyInstaller assigns __file__ inside the extraction root even when src
    # lives only in the module archive, not as a physical directory on disk.
    monkeypatch.setattr(utils, "__file__", str(bundle / "src" / "utils.py"))
    elsewhere = tmp_path / "unrelated working directory"
    elsewhere.mkdir()
    (elsewhere / "config.yaml").write_text("version: wrong-config\n", encoding="utf-8")
    monkeypatch.chdir(elsewhere)
    home = tmp_path / "PeopleOS User Data"
    if use_peopleos_home:
        monkeypatch.setenv("PEOPLEOS_HOME", str(home))
    else:
        monkeypatch.delenv("PEOPLEOS_HOME", raising=False)

    config = load_config()

    assert config["version"] == "bundle-fixture"
    expected_database = str(home / "data" / "peopleos.db") if use_peopleos_home else "data/peopleos.db"
    assert config["persistence"]["database_path"] == expected_database
    assert (bundle / "config.yaml").read_text(encoding="utf-8") == config_text
    assert (bundle / "src").exists() == source_directory_exists


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


def test_dataset_artifact_preserves_snapshot_history(monkeypatch, tmp_path: Path):
    home = tmp_path / "snapshot-store"
    monkeypatch.setenv("PEOPLEOS_HOME", str(home))
    frame = pd.DataFrame({
        "EmployeeID": ["E1", "E1", "E2", "E2"],
        "SnapshotDate": ["2026-01-31", "2026-06-30", "2026-01-31", "2026-06-30"],
        "Dept": ["A", "A", "B", "B"],
        "Salary": [50_000, 52_000, 60_000, 62_000],
    })
    path = save_dataset_artifact("ds_test", frame)
    restored = load_dataset_artifact("ds_test")
    assert path == home.resolve() / "datasets" / "ds_test.csv"
    assert restored is not None
    assert len(restored) == 4
    assert restored["EmployeeID"].tolist() == frame["EmployeeID"].tolist()
    assert restored["SnapshotDate"].tolist() == frame["SnapshotDate"].tolist()
