"""Prepare PeopleOS local AI through the installed Ollama runtime.

This explicit operator script starts Ollama when needed, chooses or downloads
one local model, runs a no-workforce-data smoke test, and persists the owner's
choice. If the PeopleOS API is already running, it refreshes the in-memory
runtime as well; otherwise restart the API after the script completes.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.platform.ai_runtime import LocalLLMSetupError, prepare_local_ollama  # noqa: E402


def _refresh_api(api_url: str, model: str) -> tuple[bool, str]:
    request = urllib.request.Request(
        f"{api_url.rstrip('/')}/api/llm/configure",
        data=json.dumps({"provider": "ollama", "enabled": True, "model": model}).encode("utf-8"),
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return True, response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return False, f"PeopleOS API returned HTTP {exc.code}; restart the API to load the saved preference."
    except (OSError, urllib.error.URLError, TimeoutError):
        return False, "PeopleOS API is not running; start or restart it to load the saved preference."


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Local Ollama model name; if omitted, use the first preferred installed model or the configured default.")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000", help="PeopleOS API URL to refresh after setup.")
    parser.add_argument("--no-api-refresh", action="store_true", help="Only prepare Ollama and persist preferences; do not contact PeopleOS API.")
    args = parser.parse_args()

    def progress(percent: int, message: str) -> None:
        print(f"[{percent:>3}%] {message}")

    try:
        result = prepare_local_ollama(args.model or "", progress)
    except LocalLLMSetupError as exc:
        print(f"Local AI setup failed: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Local AI setup failed unexpectedly: {type(exc).__name__}", file=sys.stderr)
        return 3

    refreshed = False
    refresh_message = "API refresh skipped."
    if not args.no_api_refresh:
        refreshed, refresh_message = _refresh_api(args.api_url, result["model"])
    print(json.dumps({
        "status": "ready",
        "model": result["model"],
        "smoke_test": result["smoke_test"],
        "api_refreshed": refreshed,
        "message": refresh_message,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
