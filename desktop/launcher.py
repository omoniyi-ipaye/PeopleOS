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
import traceback
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


def _record_smoke_failure() -> None:
    if os.getenv("PEOPLEOS_SMOKE_TEST") != "1":
        return
    details = traceback.format_exc()
    try:
        smoke_home = Path(os.getenv("PEOPLEOS_HOME") or Path.cwd())
        smoke_home.mkdir(parents=True, exist_ok=True)
        (smoke_home / "desktop-smoke-failure.log").write_text(details, encoding="utf-8")
    except Exception:
        pass
    if sys.stderr is not None:
        try:
            print(details, file=sys.stderr, flush=True)
        except Exception:
            pass


def main() -> int:
    port, ui_dir = _configure_environment()
    if not ui_dir.exists():
        raise RuntimeError("PeopleOS UI assets are missing. Build with PEOPLEOS_DESKTOP_BUILD=1 before packaging.")

    import uvicorn
    from api.main import app
    from desktop.control import controller
    from desktop.static_ui import install_static_ui

    install_static_ui(app, ui_dir)
    _restore_workspace()

    health_url = f"http://127.0.0.1:{port}/api/health"
    app_url = f"http://127.0.0.1:{port}/"
    first_start = True

    while True:
        config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", access_log=False)
        server = uvicorn.Server(config)
        controller.configure(
            app_url=app_url,
            open_callback=lambda: webbrowser.open(app_url, new=0, autoraise=True),
            stop_callback=lambda: setattr(server, "should_exit", True),
        )
        thread = threading.Thread(target=server.run, name="peopleos-local-server", daemon=True)
        thread.start()
        _wait_until_ready(health_url)

        if os.getenv("PEOPLEOS_SMOKE_TEST") == "1":
            with urllib.request.urlopen(app_url, timeout=3.0) as response:
                body = response.read(4096).decode("utf-8", errors="ignore")
                if response.status != 200 or "PeopleOS" not in body:
                    raise RuntimeError("Packaged PeopleOS UI smoke test failed")
            # Desktop controls must exist in the packaged build without exposing a
            # generic process surface.
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/desktop/status", timeout=3.0) as response:
                status_body = response.read().decode("utf-8", errors="ignore")
                if response.status != 200 or '"desktop":true' not in status_body.replace(" ", "").lower():
                    raise RuntimeError("Packaged PeopleOS desktop control smoke test failed")
            server.should_exit = True
            thread.join(timeout=5)
            return 0

        # First start and an intentional restart both reopen the product. A user
        # never needs to discover the local port or find a hidden process.
        webbrowser.open(app_url, new=1 if first_start else 0, autoraise=True)
        first_start = False

        try:
            while thread.is_alive():
                time.sleep(0.25)
        except KeyboardInterrupt:
            server.should_exit = True
            thread.join(timeout=5)
            return 0

        action = controller.consume_action()
        if action == "restart":
            thread.join(timeout=5)
            continue
        # Explicit Quit or an unexpected server stop both end the packaged app.
        server.should_exit = True
        thread.join(timeout=5)
        return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException:
        _record_smoke_failure()
        raise SystemExit(1)
