"""Durable local owner lock for the PeopleOS installation.

The lock is deliberately a small local boundary: it keeps an unattended
PeopleOS session from exposing workforce evidence, while the operating-system
account and full-disk protection remain the stronger security boundary.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from src.local_paths import get_peopleos_paths


_LOCK = threading.RLock()
_SCHEMA_VERSION = 1
_PBKDF2_ITERATIONS = 310_000


class AppLockError(ValueError):
    """Raised when a local app-lock transition is invalid."""


def _validate_pin(pin: str) -> str:
    if not isinstance(pin, str) or len(pin) != 6 or not pin.isascii() or not pin.isdecimal():
        raise AppLockError("The owner PIN must contain exactly six digits.")
    return pin


def _hash_pin(pin: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", pin.encode("ascii"), salt, _PBKDF2_ITERATIONS)


class AppLockStore:
    """Atomically persisted app-lock state scoped to the PeopleOS data home."""

    def __init__(self, path: Optional[str] = None):
        default_path = get_peopleos_paths().config / "app-lock.json"
        self.path = Path(path or default_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _read(self) -> Dict[str, Any]:
        with _LOCK:
            if not self.path.exists():
                return {"schema_version": _SCHEMA_VERSION, "enabled": False, "locked": False}
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise RuntimeError("PeopleOS app-lock configuration could not be read safely.") from exc
            if not isinstance(payload, dict) or payload.get("schema_version") != _SCHEMA_VERSION:
                raise RuntimeError("PeopleOS app-lock configuration has an unsupported schema.")
            enabled = payload.get("enabled")
            locked = payload.get("locked")
            if not isinstance(enabled, bool) or not isinstance(locked, bool):
                raise RuntimeError("PeopleOS app-lock configuration is malformed.")
            if enabled:
                if not isinstance(payload.get("salt"), str) or not isinstance(payload.get("pin_hash"), str):
                    raise RuntimeError("PeopleOS app-lock configuration is incomplete.")
                try:
                    salt = base64.b64decode(payload["salt"], validate=True)
                    pin_hash = base64.b64decode(payload["pin_hash"], validate=True)
                except (ValueError, TypeError) as exc:
                    raise RuntimeError("PeopleOS app-lock configuration is malformed.") from exc
                if len(salt) < 16 or len(pin_hash) != hashlib.sha256().digest_size:
                    raise RuntimeError("PeopleOS app-lock configuration is malformed.")
            return payload

    def _write(self, payload: Dict[str, Any]) -> None:
        with _LOCK:
            descriptor, tmp_name = tempfile.mkstemp(
                prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
            )
            tmp = Path(tmp_name)
            try:
                os.fchmod(descriptor, 0o600)
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle, indent=2, sort_keys=True)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(tmp, self.path)
                try:
                    self.path.chmod(0o600)
                except OSError:
                    pass
            finally:
                if tmp.exists():
                    tmp.unlink()

    def status(self) -> Dict[str, bool]:
        payload = self._read()
        enabled = bool(payload["enabled"])
        return {"enabled": enabled, "locked": bool(enabled and payload["locked"])}

    def setup(self, pin: str) -> Dict[str, bool]:
        pin = _validate_pin(pin)
        with _LOCK:
            payload = self._read()
            if payload["enabled"]:
                raise AppLockError("An owner PIN is already configured. Use the existing PIN to unlock PeopleOS.")
            salt = secrets.token_bytes(16)
            payload = {
                "schema_version": _SCHEMA_VERSION,
                "enabled": True,
                "locked": False,
                "salt": base64.b64encode(salt).decode("ascii"),
                "pin_hash": base64.b64encode(_hash_pin(pin, salt)).decode("ascii"),
            }
            self._write(payload)
            return self.status()

    def lock(self) -> Dict[str, bool]:
        with _LOCK:
            payload = self._read()
            if not payload["enabled"]:
                raise AppLockError("Set an owner PIN before locking PeopleOS.")
            payload["locked"] = True
            self._write(payload)
            return self.status()

    def unlock(self, pin: str) -> Dict[str, bool]:
        pin = _validate_pin(pin)
        with _LOCK:
            payload = self._read()
            self._require_current_pin(payload, pin)
            payload["locked"] = False
            self._write(payload)
            return self.status()

    @staticmethod
    def _require_current_pin(payload: Dict[str, Any], pin: str) -> None:
        if not payload["enabled"]:
            raise AppLockError("No owner PIN is configured.")
        salt = base64.b64decode(payload["salt"], validate=True)
        expected = base64.b64decode(payload["pin_hash"], validate=True)
        if not hmac.compare_digest(_hash_pin(pin, salt), expected):
            raise AppLockError("The current owner PIN is incorrect.")

    def change_pin(self, current_pin: str, new_pin: str) -> Dict[str, bool]:
        current_pin = _validate_pin(current_pin)
        new_pin = _validate_pin(new_pin)
        with _LOCK:
            payload = self._read()
            self._require_current_pin(payload, current_pin)
            salt = secrets.token_bytes(16)
            payload["salt"] = base64.b64encode(salt).decode("ascii")
            payload["pin_hash"] = base64.b64encode(_hash_pin(new_pin, salt)).decode("ascii")
            self._write(payload)
            return self.status()

    def disable(self, current_pin: str) -> Dict[str, bool]:
        current_pin = _validate_pin(current_pin)
        with _LOCK:
            payload = self._read()
            self._require_current_pin(payload, current_pin)
            self._write({"schema_version": _SCHEMA_VERSION, "enabled": False, "locked": False})
            return self.status()

    def is_locked(self) -> bool:
        return self.status()["locked"]
