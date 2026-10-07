"""Regression tests for the local owner app lock."""

import json

import pytest
from fastapi.testclient import TestClient

from src.platform.app_lock import AppLockError, AppLockStore


def test_app_lock_persists_hashed_pin_and_survives_reopen(tmp_path):
    path = tmp_path / "config" / "app-lock.json"
    first = AppLockStore(str(path))

    assert first.status() == {"enabled": False, "locked": False}
    assert first.setup("012345") == {"enabled": True, "locked": False}
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert "012345" not in path.read_text(encoding="utf-8")
    assert stored["enabled"] is True
    assert stored["locked"] is False

    first.lock()
    reopened = AppLockStore(str(path))
    assert reopened.status() == {"enabled": True, "locked": True}
    with pytest.raises(AppLockError, match="current owner PIN"):
        reopened.unlock("012346")
    assert reopened.unlock("012345") == {"enabled": True, "locked": False}


@pytest.mark.parametrize("pin", ["12345", "1234567", "12a456", "１２３４５６"])
def test_app_lock_requires_six_ascii_digits(tmp_path, pin):
    with pytest.raises(AppLockError, match="exactly six digits"):
        AppLockStore(str(tmp_path / "app-lock.json")).setup(pin)


def test_app_lock_cannot_be_reconfigured_without_a_new_installation(tmp_path):
    store = AppLockStore(str(tmp_path / "app-lock.json"))
    store.setup("123456")
    with pytest.raises(AppLockError, match="already configured"):
        store.setup("654321")


def test_app_lock_owner_can_change_or_remove_pin_with_current_pin(tmp_path):
    store = AppLockStore(str(tmp_path / "app-lock.json"))
    store.setup("123456")
    with pytest.raises(AppLockError, match="current owner PIN"):
        store.change_pin("000000", "654321")
    assert store.change_pin("123456", "654321") == {"enabled": True, "locked": False}
    with pytest.raises(AppLockError, match="current owner PIN"):
        store.unlock("123456")
    store.lock()
    assert store.unlock("654321") == {"enabled": True, "locked": False}
    with pytest.raises(AppLockError, match="current owner PIN"):
        store.disable("123456")
    assert store.disable("654321") == {"enabled": False, "locked": False}


def test_locked_local_api_fails_closed_until_owner_unlocks(monkeypatch, tmp_path):
    monkeypatch.setenv("PEOPLEOS_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("PEOPLEOS_WORKSPACE_REGISTRY", str(tmp_path / "registry.json"))
    from api.main import app

    with TestClient(app, base_url="http://testserver") as client:
        assert client.get("/api/app-lock/status").json() == {"enabled": False, "locked": False}
        assert client.post("/api/app-lock/setup", json={"pin": "123456"}).status_code == 200
        assert client.post("/api/app-lock/lock").json() == {"enabled": True, "locked": True}
        blocked = client.get("/api/status")
        assert blocked.status_code == 423
        assert blocked.json()["code"] == "app_locked"
        assert client.post("/api/app-lock/unlock", json={"pin": "123456"}).json() == {"enabled": True, "locked": False}
        assert client.post("/api/app-lock/change", json={"current_pin": "123456", "new_pin": "654321"}).json() == {"enabled": True, "locked": False}
        assert client.post("/api/app-lock/disable", json={"pin": "654321"}).json() == {"enabled": False, "locked": False}
