"""In-process lifecycle control for the packaged PeopleOS desktop runtime.

The controller is intentionally tiny: it exposes only open/restart/quit actions
registered by the one-click launcher. It never runs shell commands and remains
inactive outside PEOPLEOS_DESKTOP mode.
"""
from __future__ import annotations

import os
import threading
from typing import Callable, Optional


class DesktopLifecycleController:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._action: Optional[str] = None
        self._app_url: Optional[str] = None
        self._open_callback: Optional[Callable[[], None]] = None
        self._stop_callback: Optional[Callable[[], None]] = None

    @property
    def enabled(self) -> bool:
        return os.getenv("PEOPLEOS_DESKTOP") == "1"

    def configure(self, *, app_url: str, open_callback: Callable[[], None], stop_callback: Callable[[], None]) -> None:
        with self._lock:
            self._app_url = app_url
            self._open_callback = open_callback
            self._stop_callback = stop_callback
            self._action = None

    def status(self) -> dict[str, object]:
        with self._lock:
            return {
                "desktop": self.enabled,
                "app_url": self._app_url if self.enabled else None,
                "restart_supported": self.enabled and self._stop_callback is not None,
                "quit_supported": self.enabled and self._stop_callback is not None,
            }

    def open(self) -> None:
        if not self.enabled or self._open_callback is None:
            raise RuntimeError("Desktop controls are not available in this runtime.")
        self._open_callback()

    def request(self, action: str) -> None:
        if action not in {"restart", "quit"}:
            raise ValueError("Unsupported desktop lifecycle action.")
        if not self.enabled or self._stop_callback is None:
            raise RuntimeError("Desktop controls are not available in this runtime.")
        with self._lock:
            self._action = action
        self._stop_callback()

    def consume_action(self) -> Optional[str]:
        with self._lock:
            action = self._action
            self._action = None
            return action


controller = DesktopLifecycleController()
