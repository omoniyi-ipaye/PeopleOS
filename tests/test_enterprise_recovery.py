"""Exact-byte preservation and explicit failure semantics for metadata recovery."""
import hashlib
import json
from pathlib import Path

import pytest

from src.platform.health import SystemHealthMonitor
from src.platform.workspace import WorkspaceStore


def monitor_with_corruption(tmp_path, content):
    store = WorkspaceStore(str(tmp_path / 'registry.json'))
    store.path.write_bytes(content)
    return SystemHealthMonitor(store)


@pytest.mark.parametrize('original', [b'{"workspaces": [broken\xff', b'{"schema_version":1,"workspaces":[{"not":"a workspace"}]}', b'{"schema_version":99,"workspaces":[]}'])
def test_recovery_quarantines_exact_bytes_without_reactivating_data(tmp_path, original):
    monitor = monitor_with_corruption(tmp_path, original)
    result = monitor.recover()
    backup = tmp_path / result['quarantine']['filename']
    assert backup.read_bytes() == original
    assert result['quarantine']['sha256'] == hashlib.sha256(original).hexdigest()
    assert result['status'] == 'metadata_reinitialized'
    assert result['requires_review'] is True
    current = monitor.store.get_workspace('local')
    assert current.active_dataset_id is None
    assert current.active_model_id is None
    assert current.datasets == []
    assert result['health']['status'] == 'degraded'
    assert SystemHealthMonitor(monitor.store).check()['recovery_required'] is True


def test_quarantine_write_failure_retains_original_registry(tmp_path, monkeypatch):
    original = b'broken original registry'
    monitor = monitor_with_corruption(tmp_path, original)
    real_open = __import__('os').open
    def fail_backup(path, *args, **kwargs):
        if '.quarantine-' in str(path):
            raise OSError('disk unavailable')
        return real_open(path, *args, **kwargs)
    monkeypatch.setattr('src.platform.health.os.open', fail_backup)
    with pytest.raises(OSError, match='disk unavailable'):
        monitor.recover()
    assert monitor.store.path.read_bytes() == original


def test_repeated_recovery_never_overwrites_previous_quarantine(tmp_path):
    monitor = monitor_with_corruption(tmp_path, b'first original')
    first = monitor.recover()
    monitor.store.path.write_bytes(b'second original')
    second = monitor.recover()
    assert first['quarantine']['filename'] != second['quarantine']['filename']
    assert (tmp_path / first['quarantine']['filename']).read_bytes() == b'first original'
    assert (tmp_path / second['quarantine']['filename']).read_bytes() == b'second original'


def test_registry_change_during_quarantine_aborts_replacement(tmp_path, monkeypatch):
    monitor = monitor_with_corruption(tmp_path, b'original')
    backup = monitor._quarantine_registry
    def racing_backup(content):
        result = backup(content)
        monitor.store.path.write_bytes(b'new version must survive')
        return result
    monkeypatch.setattr(monitor, '_quarantine_registry', racing_backup)
    with pytest.raises(RuntimeError, match='changed during recovery'):
        monitor.recover()
    assert monitor.store.path.read_bytes() == b'new version must survive'
    assert list(tmp_path.glob('*.quarantine-*'))[0].read_bytes() == b'original'


def test_valid_registry_failure_is_not_reported_as_success(tmp_path, monkeypatch):
    monitor = SystemHealthMonitor(WorkspaceStore(str(tmp_path / 'registry.json')))
    before = monitor.store.path.read_bytes()
    monkeypatch.setattr(monitor.store, 'ensure_workspace', lambda *args: (_ for _ in ()).throw(OSError('cannot update')))
    with pytest.raises(OSError, match='cannot update'):
        monitor.recover()
    assert monitor.store.path.read_bytes() == before
    assert list(tmp_path.glob('*.quarantine-*')) == []


def test_valid_recovery_keeps_registered_dataset_and_model_history(tmp_path):
    store = WorkspaceStore(str(tmp_path / 'registry.json'))
    dataset = store.register_dataset(workspace_id='local', source_name='synthetic.csv', content_hash='a'*64, row_count=100, columns=['EmployeeID'])
    store.activate_dataset('local', dataset.dataset_id)
    before = store.path.read_bytes()
    result = SystemHealthMonitor(store).recover()
    assert store.path.read_bytes() == before
    assert result['status'] == 'healthy'
    assert result['requires_review'] is False
    assert result['quarantine'] is None


@pytest.mark.parametrize('original', [b'{broken json', b'null', b'{"schema_version":1,"workspaces":[{"bad":"record"}]}', b'{"schema_version":2,"workspaces":[]}'])
def test_actual_constructor_preserves_malformed_existing_registry(tmp_path, original):
    registry = tmp_path / 'registry.json'
    registry.write_bytes(original)
    with pytest.raises(RuntimeError, match='original bytes were retained'):
        WorkspaceStore(str(registry))
    assert registry.read_bytes() == original
    assert list(tmp_path.iterdir()) == [registry]


def test_missing_registry_initializes_but_missing_live_registry_fails_closed(tmp_path):
    registry = tmp_path / 'registry.json'
    store = WorkspaceStore(str(registry))
    assert store.get_workspace('local').workspace_id == 'local'
    registry.unlink()
    with pytest.raises(RuntimeError, match='missing or invalid'):
        store.ensure_workspace('new')
    assert not registry.exists()


def test_recovery_stays_degraded_until_explicit_dataset_activation(tmp_path):
    monitor = monitor_with_corruption(tmp_path, b'broken registry')
    monitor.recover()
    # Reopening a repaired registry cannot erase the required reconciliation.
    reopened = WorkspaceStore(str(monitor.store.path))
    assert SystemHealthMonitor(reopened).check()['status'] == 'degraded'
    dataset = reopened.register_dataset(workspace_id='local', source_name='synthetic.csv', content_hash='b'*64, row_count=100, columns=['EmployeeID'])
    assert SystemHealthMonitor(reopened).check()['status'] == 'degraded'
    reopened.activate_dataset('local', dataset.dataset_id)
    assert SystemHealthMonitor(reopened).check()['status'] == 'healthy'
    assert reopened._read()['recovery_quarantine'] is not None
