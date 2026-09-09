"""Regression tests for packaged desktop lifecycle controls."""
import asyncio

import pytest
from fastapi import HTTPException

from api.routes import desktop as desktop_routes
from desktop.control import controller


def test_desktop_controls_are_hidden_outside_packaged_runtime(monkeypatch):
    monkeypatch.delenv("PEOPLEOS_DESKTOP", raising=False)
    with pytest.raises(HTTPException) as exc:
        desktop_routes._require_desktop()
    assert exc.value.status_code == 404
    assert controller.status()["desktop"] is False


def test_desktop_controller_allows_only_open_restart_and_quit(monkeypatch):
    monkeypatch.setenv("PEOPLEOS_DESKTOP", "1")
    opened = []
    stopped = []
    controller.configure(app_url="http://127.0.0.1:9999/", open_callback=lambda: opened.append(True), stop_callback=lambda: stopped.append(True))

    controller.open()
    assert opened == [True]

    controller.request("restart")
    assert stopped == [True]
    assert controller.consume_action() == "restart"

    controller.request("quit")
    assert stopped == [True, True]
    assert controller.consume_action() == "quit"

    with pytest.raises(ValueError):
        controller.request("shell")


def test_desktop_status_reports_only_bounded_capabilities(monkeypatch):
    monkeypatch.setenv("PEOPLEOS_DESKTOP", "1")
    controller.configure(app_url="http://127.0.0.1:9999/", open_callback=lambda: None, stop_callback=lambda: None)
    status = asyncio.run(desktop_routes.desktop_status())
    assert status == {
        "desktop": True,
        "app_url": "http://127.0.0.1:9999/",
        "restart_supported": True,
        "quit_supported": True,
    }
    assert not {"command", "shell", "path", "process"} & set(status)
