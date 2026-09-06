"""One-click launcher for the local-first PeopleOS product.

The packaged application contains the FastAPI runtime and a statically exported
PeopleOS UI. It binds only to loopback, restores the user's durable local
workspace, opens the product automatically, and requires no Python, Node,
Docker, or command-line setup on the user's machine.
"""

from __future__ import annotations

import os
import socket
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path


def _bundle_root() -> Path:
    return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _configure_environment() -> tuple[int, Path]:
    # Import only the path helper before the API so every persistence consumer
    # observes the correct PEOPLEOS_HOME from its first initialization.
    from src.local_paths import get_peopleos_paths

    paths = get_peopleos_paths()
    ui_dir = _bundle_root() / "peopleos_ui"
    if not ui_dir.exists():
        source_ui = Path(__file__).resolve().parents[1] / "web" / "out"
        ui_dir = source_ui if source_ui.exists() else ui_dir

    port = _find_free_port()
    os.environ["PEOPLEOS_HOME"] = str(paths.root)
    os.environ["PEOPLEOS_DESKTOP"] = "1"
    os.environ["PEOPLEOS_UI_DIR"] = str(ui_dir)
    os.environ["PEOPLEOS_API_HOST"] = "127.0.0.1"
    os.environ["PEOPLEOS_API_PORT"] = str(port)
    return port, ui_dir


def _restore_workspace() -> None:
    try:
        from api.runtime_registry import get_local_state

        get_local_state().load_from_database()
    except Exception:
        # First run and empty databases are normal. The product remains usable
        # and will present the add-data onboarding state.
        return


def _wait_until_ready(url: str, timeout_seconds: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as response:
                if response.status == 200:
                    return
        except Exception as exc:  # pragma: no cover - timing dependent
            last_error = exc
            time.sleep(0.15)
    raise RuntimeError(f"PeopleOS did not become ready: {last_error}")


def main() -> int:
    port, ui_dir = _configure_environment()
    if not ui_dir.exists():
        raise RuntimeError(
            "PeopleOS UI assets are missing. Build with PEOPLEOS_DESKTOP_BUILD=1 before packaging."
        )

    # Import after environment configuration so runtime paths are deterministic.
    import uvicorn
    from api.main import app
    from desktop.static_ui import install_static_ui

    install_static_ui(app, ui_dir)
    _restore_workspace()

    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, name="peopleos-local-server", daemon=True)
    thread.start()

    health_url = f"http://127.0.0.1:{port}/api/health"
    app_url = f"http://127.0.0.1:{port}/"
    _wait_until_ready(health_url)
    webbrowser.open(app_url, new=1, autoraise=True)

    # Keep the local service alive while the launcher process is running.
    # Packaged desktop builds use a windowless process on Windows/macOS.
    try:
        while thread.is_alive():
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.should_exit = True
        thread.join(timeout=5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
