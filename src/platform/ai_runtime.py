"""Local AI preference, Ollama discovery and bounded setup helpers.

PeopleOS keeps deterministic analysis independent from model availability. This
module owns the optional local Ollama setup path so onboarding can offer a clear
owner-controlled switch without embedding a model, starting a background
service unexpectedly, or sending workforce data to a remote provider.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from src.local_paths import get_peopleos_paths
from src.logger import get_logger
from src.utils import load_config


logger = get_logger("ai_runtime")

_SCHEMA_VERSION = 1
_MODEL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
_DEFAULT_HOST = "http://127.0.0.1:11434"
_SETUP_MODEL = "gemma3:4b"
_INSTALL_GUIDE = "https://ollama.com/download"
_PREFERRED_INSTALLED_MODELS = (
    "qwen3.8:latest",
    "hermes3:latest",
    "gemma3:4b",
    "llama3.2:3b",
    "qwen3:4b",
)


class LocalLLMSetupError(RuntimeError):
    """A user-actionable local model setup failure."""

    def __init__(self, message: str, *, code: str = "setup_failed"):
        super().__init__(message)
        self.code = code


def _configured_defaults() -> tuple[str, str]:
    try:
        config = load_config()
        ollama = config.get("ollama", {}) if isinstance(config, dict) else {}
        host = ollama.get("host", _DEFAULT_HOST)
        model = ollama.get("model", _SETUP_MODEL)
        return _validate_host(str(host)), _validate_local_model(str(model))
    except Exception:
        return _DEFAULT_HOST, _SETUP_MODEL


def _validate_model(model: str) -> str:
    value = model.strip()
    if not value or not _MODEL_PATTERN.fullmatch(value):
        raise LocalLLMSetupError(
            "Choose a valid local Ollama model name, for example gemma3:4b.",
            code="invalid_model",
        )
    return value


def _validate_host(host: str) -> str:
    value = host.strip().rstrip("/")
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in _LOCAL_HOSTS:
        raise LocalLLMSetupError(
            "PeopleOS only supports an Ollama server on this computer for local AI.",
            code="non_local_host",
        )
    return value


def canonical_model_name(model: str) -> str:
    value = _validate_model(model)
    last_segment = value.rsplit("/", 1)[-1]
    return value if ":" in last_segment else f"{value}:latest"


def _validate_local_model(model: str) -> str:
    """Accept only models whose weights are hosted by this Ollama instance."""
    canonical = canonical_model_name(model)
    if canonical.endswith(":cloud"):
        raise LocalLLMSetupError(
            "PeopleOS local AI needs a model downloaded to this computer; cloud models are not allowed here.",
            code="remote_model_not_allowed",
        )
    return canonical


class AIPreferencesStore:
    """Atomically persisted owner preference for optional local AI."""

    def __init__(self, path: Optional[str] = None):
        default_path = get_peopleos_paths().config / "ai-preferences.json"
        self.path = Path(path or default_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _default(self) -> Dict[str, Any]:
        host, model = _configured_defaults()
        return {
            "schema_version": _SCHEMA_VERSION,
            "provider": "none",
            "enabled": False,
            "ollama_host": host,
            "ollama_model": model,
        }

    def _read(self) -> Optional[Dict[str, Any]]:
        if not self.path.exists():
            return None
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("PeopleOS AI preferences could not be read safely.") from exc
        if not isinstance(payload, dict) or payload.get("schema_version") != _SCHEMA_VERSION:
            raise RuntimeError("PeopleOS AI preferences have an unsupported schema.")
        provider = payload.get("provider")
        enabled = payload.get("enabled")
        if provider not in {"none", "ollama"} or not isinstance(enabled, bool):
            raise RuntimeError("PeopleOS AI preferences are malformed.")
        host = _validate_host(str(payload.get("ollama_host", _DEFAULT_HOST)))
        model = _validate_local_model(str(payload.get("ollama_model", _SETUP_MODEL)))
        return {
            "schema_version": _SCHEMA_VERSION,
            "provider": provider,
            "enabled": enabled and provider == "ollama",
            "ollama_host": host,
            "ollama_model": model,
        }

    def get(self) -> Dict[str, Any]:
        return self._read() or self._default()

    def is_configured(self) -> bool:
        return self.path.exists()

    def _write(self, payload: Dict[str, Any]) -> None:
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

    def update(self, *, enabled: bool, model: Optional[str] = None, provider: str = "ollama") -> Dict[str, Any]:
        if provider not in {"none", "ollama"}:
            raise LocalLLMSetupError("PeopleOS supports local Ollama or deterministic mode.", code="invalid_provider")
        current = self.get()
        selected_model = _validate_local_model(model or current["ollama_model"])
        payload = {
            "schema_version": _SCHEMA_VERSION,
            "provider": "ollama" if enabled else "none",
            "enabled": bool(enabled and provider == "ollama"),
            "ollama_host": current["ollama_host"],
            "ollama_model": selected_model,
        }
        self._write(payload)
        return payload


def find_ollama_binary() -> Optional[str]:
    """Find the official Ollama executable without modifying the machine."""
    found = shutil.which("ollama")
    if found:
        return found
    candidates: list[Path] = []
    if sys.platform == "darwin":
        candidates.extend([
            Path("/Applications/Ollama.app/Contents/Resources/ollama"),
            Path("/opt/homebrew/bin/ollama"),
            Path("/usr/local/bin/ollama"),
        ])
    elif os.name == "nt":
        for variable in ("LOCALAPPDATA", "PROGRAMFILES"):
            base = os.getenv(variable)
            if base:
                candidates.append(Path(base) / "Programs" / "Ollama" / "ollama.exe")
                candidates.append(Path(base) / "Ollama" / "ollama.exe")
    else:
        candidates.extend([Path("/usr/local/bin/ollama"), Path("/usr/bin/ollama")])
    return next((str(path) for path in candidates if path.is_file() and os.access(path, os.X_OK)), None)


def _request_json(host: str, path: str, *, method: str = "GET", payload: Optional[dict] = None, timeout: float = 5.0) -> dict:
    base = _validate_host(host)
    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(f"{base}{path}", data=body, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            content = response.read()
    except HTTPError as exc:
        raise LocalLLMSetupError(f"Ollama returned HTTP {exc.code} while contacting {path}.", code="ollama_http_error") from exc
    except (OSError, URLError, TimeoutError) as exc:
        raise LocalLLMSetupError("Ollama is not running or is not reachable on this computer.", code="ollama_unreachable") from exc
    try:
        result = json.loads(content.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise LocalLLMSetupError("Ollama returned an invalid response.", code="ollama_invalid_response") from exc
    if not isinstance(result, dict):
        raise LocalLLMSetupError("Ollama returned an invalid response.", code="ollama_invalid_response")
    return result


def generate_ollama(
    host: str,
    model: str,
    prompt: str,
    *,
    options: Optional[dict[str, Any]] = None,
    response_format: Optional[str] = None,
    timeout: float = 180.0,
) -> dict[str, Any]:
    """Generate through Ollama's local HTTP API with thinking disabled.

    The repository's pinned Python Ollama client predates the API's ``think``
    flag. Calling the local HTTP endpoint here keeps newer thinking models from
    returning an empty ``response`` while preserving the existing client API.
    """
    payload: dict[str, Any] = {
        "model": _validate_local_model(model),
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": dict(options or {}),
    }
    if response_format:
        payload["format"] = response_format
    return _request_json(host, "/api/generate", method="POST", payload=payload, timeout=timeout)


def _model_records(payload: dict) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in payload.get("models", []):
        if not isinstance(item, dict):
            continue
        name = item.get("model") or item.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        try:
            canonical = canonical_model_name(name)
        except LocalLLMSetupError:
            continue
        capabilities = item.get("capabilities")
        if isinstance(capabilities, list) and "completion" not in capabilities:
            # Embedding-only models belong to semantic search, not local AI
            # synthesis. Do not offer them in the onboarding model picker.
            continue
        records.append({
            "name": canonical,
            "digest": item.get("digest"),
            "size": item.get("size") if isinstance(item.get("size"), int) else None,
            "remote": canonical.endswith(":cloud"),
        })
    return sorted(records, key=lambda item: item["name"])


def _choose_model(selected: str, records: list[dict[str, Any]]) -> str:
    available = {item["name"] for item in records if not item.get("remote")}
    canonical = canonical_model_name(selected)
    if canonical in available:
        return canonical
    for preferred in _PREFERRED_INSTALLED_MODELS:
        if preferred in available:
            return preferred
    if available:
        return sorted(available)[0]
    if canonical.endswith(":cloud"):
        return _SETUP_MODEL
    return canonical


def probe_ollama(host: str, selected_model: str) -> dict[str, Any]:
    """Return local installation and model readiness without changing state."""
    host = _validate_host(host)
    selected = _validate_local_model(selected_model)
    binary = find_ollama_binary()
    try:
        payload = _request_json(host, "/api/tags", timeout=3.0)
        records = _model_records(payload)
        installed = next((item for item in records if item["name"] == selected), None)
        local_records = [item for item in records if not item.get("remote")]
        recommended = _choose_model(selected, records)
        return {
            "ollama_installed": binary is not None,
            "ollama_binary": binary,
            "ollama_running": True,
            "host": host,
            "selected_model": selected,
            "selected_model_installed": installed is not None and not installed.get("remote", False),
            "selected_model_digest": installed.get("digest") if installed else None,
            "installed_models": local_records,
            "recommended_model": recommended,
            "reason": None if installed and not installed.get("remote", False) else (
                f"Model {selected} is not installed locally." if local_records else "Ollama is ready, but no local model is installed."
            ),
        }
    except LocalLLMSetupError as exc:
        return {
            "ollama_installed": binary is not None,
            "ollama_binary": binary,
            "ollama_running": False,
            "host": host,
            "selected_model": selected,
            "selected_model_installed": False,
            "selected_model_digest": None,
            "installed_models": [],
            "recommended_model": selected,
            "reason": str(exc),
        }


def _wait_for_server(host: str, selected_model: str, timeout: float = 15.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    latest = probe_ollama(host, selected_model)
    while time.monotonic() < deadline:
        if latest["ollama_running"]:
            return latest
        time.sleep(0.5)
        latest = probe_ollama(host, selected_model)
    return latest


def ensure_ollama_running(host: str, selected_model: str, progress: Optional[Callable[[int, str], None]] = None) -> dict[str, Any]:
    current = probe_ollama(host, selected_model)
    if current["ollama_running"]:
        return current
    binary = current.get("ollama_binary") or find_ollama_binary()
    if not binary:
        raise LocalLLMSetupError(
            f"Ollama is not installed. Download it from {_INSTALL_GUIDE}, then run setup again.",
            code="ollama_not_installed",
        )
    if progress:
        progress(10, "Starting the local Ollama service…")
    environment = os.environ.copy()
    environment["OLLAMA_HOST"] = host
    try:
        subprocess.Popen(
            [binary, "serve"],
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError as exc:
        raise LocalLLMSetupError("PeopleOS could not start Ollama automatically. Use the download guide for manual setup.", code="ollama_start_failed") from exc
    latest = _wait_for_server(host, selected_model)
    if not latest["ollama_running"]:
        raise LocalLLMSetupError("Ollama did not become ready within 15 seconds. Check the Ollama app and try again.", code="ollama_start_timeout")
    return latest


def pull_ollama_model(host: str, model: str, progress: Optional[Callable[[int, str], None]] = None) -> None:
    """Pull one explicit model through Ollama's local streaming API."""
    selected = _validate_local_model(model)
    request = Request(
        f"{_validate_host(host)}/api/pull",
        data=json.dumps({"model": selected, "stream": True}).encode("utf-8"),
        headers={"Accept": "application/x-ndjson", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=1800) as response:
            for raw_line in response:
                if not raw_line.strip():
                    continue
                try:
                    item = json.loads(raw_line.decode("utf-8"))
                except (UnicodeError, json.JSONDecodeError) as exc:
                    raise LocalLLMSetupError("Ollama returned invalid model-download progress.", code="ollama_pull_invalid_response") from exc
                if item.get("error"):
                    raise LocalLLMSetupError(str(item["error"]), code="ollama_pull_failed")
                total = item.get("total")
                completed = item.get("completed")
                if isinstance(total, int) and total > 0 and isinstance(completed, int):
                    percent = min(95, max(15, int(completed / total * 80) + 15))
                else:
                    percent = 20
                if progress:
                    progress(percent, str(item.get("status") or f"Downloading {selected}…"))
    except HTTPError as exc:
        raise LocalLLMSetupError(f"Ollama could not download {selected} (HTTP {exc.code}).", code="ollama_pull_failed") from exc
    except (OSError, URLError, TimeoutError) as exc:
        raise LocalLLMSetupError(f"Ollama could not download {selected}.", code="ollama_pull_failed") from exc


def test_ollama_model(host: str, model: str) -> dict[str, Any]:
    """Run a no-workforce-data smoke test against the selected local model."""
    started = time.monotonic()
    result = _request_json(
        host,
        "/api/generate",
        method="POST",
        payload={
            "model": _validate_local_model(model),
            "prompt": "Reply with exactly READY and nothing else.",
            "stream": False,
            "think": False,
            "options": {"temperature": 0, "num_predict": 8},
        },
        timeout=180.0,
    )
    response = result.get("response")
    if not isinstance(response, str) or not response.strip():
        raise LocalLLMSetupError("The model responded without usable text.", code="ollama_model_empty_response")
    return {
        "passed": "ready" in response.lower(),
        "model": result.get("model") or canonical_model_name(model),
        "response": response.strip(),
        "elapsed_ms": round((time.monotonic() - started) * 1000, 1),
    }


def prepare_local_ollama(model: str, progress: Optional[Callable[[int, str], None]] = None) -> dict[str, Any]:
    """Start Ollama, pull one model if necessary, and run a local smoke test."""
    preferences = AIPreferencesStore().get()
    host = preferences["ollama_host"]
    requested = _validate_local_model(model or preferences["ollama_model"])
    current = ensure_ollama_running(host, requested, progress)
    selected = _validate_local_model(model) if model else _choose_model(requested, current.get("installed_models", []))
    installed_names = {item["name"] for item in current.get("installed_models", []) if not item.get("remote")}
    if selected not in installed_names:
        if progress:
            progress(15, f"Preparing local model {selected}…")
        pull_ollama_model(host, selected, progress)
    if progress:
        progress(96, "Testing the local model without workforce data…")
    smoke = test_ollama_model(host, selected)
    if not smoke["passed"]:
        raise LocalLLMSetupError("The local model did not pass the PeopleOS readiness test.", code="ollama_model_test_failed")
    AIPreferencesStore().update(enabled=True, model=selected)
    return {"host": host, "model": selected, "smoke_test": smoke}


class LocalLLMSetupManager:
    """One bounded background setup operation for the local onboarding action."""

    def __init__(self):
        self._lock = threading.RLock()
        self._thread: Optional[threading.Thread] = None
        self._status: Dict[str, Any] = {
            "state": "idle",
            "progress": 0,
            "message": None,
            "error_code": None,
            "started_at": None,
            "finished_at": None,
            "model": None,
        }

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._status)

    def start(self, model: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return dict(self._status)
            preferences = AIPreferencesStore().get()
            selected = _validate_local_model(model or preferences["ollama_model"])
            self._status = {
                "state": "starting",
                "progress": 1,
                "message": "Preparing local AI…",
                "error_code": None,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "finished_at": None,
                "model": canonical_model_name(selected),
            }
            self._thread = threading.Thread(target=self._run, args=(selected,), daemon=True, name="peopleos-llm-setup")
            self._thread.start()
            return dict(self._status)

    def _progress(self, percent: int, message: str) -> None:
        with self._lock:
            self._status["progress"] = percent
            self._status["message"] = message
            if percent >= 96:
                self._status["state"] = "testing"
            elif percent > 10:
                self._status["state"] = "pulling"

    def _run(self, model: str) -> None:
        try:
            result = prepare_local_ollama(model, self._progress)
            from api.runtime_registry import get_local_state
            from src.platform.ai_runtime import refresh_llm_state
            from src.platform.runtime_lock import RUNTIME_MUTATION_LOCK
            with RUNTIME_MUTATION_LOCK:
                refresh_llm_state(get_local_state())
            with self._lock:
                self._status.update({
                    "state": "ready",
                    "progress": 100,
                    "message": f"Local AI is ready with {result['model']}.",
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "model": result["model"],
                })
        except LocalLLMSetupError as exc:
            logger.warning("Local AI setup did not complete: %s", exc)
            with self._lock:
                self._status.update({
                    "state": "error",
                    "message": str(exc),
                    "error_code": exc.code,
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                })
        except Exception as exc:
            logger.exception("Unexpected local AI setup failure")
            try:
                AIPreferencesStore().update(enabled=False, model=model, provider="none")
            except Exception:
                logger.exception("Could not roll back the local AI preference after setup failure")
            with self._lock:
                self._status.update({
                    "state": "error",
                    "message": "PeopleOS could not finish local AI setup. Review the local logs and try again.",
                    "error_code": type(exc).__name__,
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                })


local_llm_setup = LocalLLMSetupManager()


def refresh_llm_state(state: Any) -> dict[str, Any]:
    """Apply the persisted local-AI choice to an already loaded runtime."""
    preferences = AIPreferencesStore().get()
    state.features_enabled = dict(getattr(state, "features_enabled", {}) or {})
    state.features_enabled["llm"] = False
    if preferences["provider"] != "ollama" or not preferences["enabled"]:
        state.llm_client = None
        if getattr(state, "raw_df", None) is not None:
            from src.insight_interpreter import InsightInterpreter
            from src.nlp_engine import NLPEngine
            state.nlp_engine = NLPEngine(None)
            state.insight_interpreter = InsightInterpreter()
        return {"enabled": False, "available": False, "model": None}

    from src.insight_interpreter import InsightInterpreter
    from src.nlp_engine import NLPEngine
    from src.safe_llm_client import SafeLLMClient

    client = SafeLLMClient(respect_preferences=True)
    state.llm_client = client
    state.features_enabled["llm"] = bool(client.is_available)
    if getattr(state, "raw_df", None) is not None:
        state.nlp_engine = NLPEngine(client)
        state.insight_interpreter = InsightInterpreter(client)
    return {
        "enabled": True,
        "available": bool(client.is_available),
        "model": client.model if client.is_available else preferences["ollama_model"],
        "reason": getattr(client, "unavailable_reason", None),
    }


def local_llm_status() -> dict[str, Any]:
    preferences = AIPreferencesStore().get()
    selected = preferences["ollama_model"]
    probe = probe_ollama(preferences["ollama_host"], selected)
    setup = local_llm_setup.status()
    ready = bool(
        preferences["enabled"]
        and preferences["provider"] == "ollama"
        and probe["ollama_running"]
        and probe["selected_model_installed"]
    )
    if ready:
        reason = None
    elif not probe["ollama_installed"]:
        reason = f"Ollama is not installed. Download it from {_INSTALL_GUIDE}."
    else:
        reason = probe["reason"]
    return {
        "provider": preferences["provider"],
        "enabled": preferences["enabled"],
        "ready": ready,
        "ollama_installed": probe["ollama_installed"],
        "ollama_running": probe["ollama_running"],
        "ollama_binary": probe["ollama_binary"],
        "host": probe["host"],
        "selected_model": probe["selected_model"],
        "selected_model_installed": probe["selected_model_installed"],
        "selected_model_digest": probe["selected_model_digest"],
        "installed_models": probe["installed_models"],
        "recommended_model": probe["recommended_model"],
        "reason": reason,
        "download_guide": _INSTALL_GUIDE,
        "setup_state": setup["state"],
        "setup_progress": setup["progress"],
        "setup_message": setup["message"],
        "setup_error_code": setup["error_code"],
        "setup_model": setup["model"],
    }
