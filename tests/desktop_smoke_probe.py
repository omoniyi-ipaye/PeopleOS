"""Launch the frozen PeopleOS desktop executable and surface startup failures.

Production desktop executables are windowed, so stdout/stderr can be unavailable
on some platforms. The launcher therefore writes a smoke-failure log under
PEOPLEOS_HOME; this probe prints both captured process output and that file so a
GitHub Actions failure always contains the underlying Python traceback.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _print_failure_log(smoke_home: Path) -> None:
    failure_log = smoke_home / "desktop-smoke-failure.log"
    if not failure_log.exists():
        print("No desktop-smoke-failure.log was produced.")
        return

    print("\n--- frozen PeopleOS startup traceback ---")
    print(failure_log.read_text(encoding="utf-8", errors="replace"))
    print("--- end frozen PeopleOS startup traceback ---\n")


def main() -> int:
    exe = Path("dist") / ("PeopleOS.exe" if os.name == "nt" else "PeopleOS")
    if not exe.exists():
        print(f"Packaged executable not found: {exe}", file=sys.stderr)
        return 2

    smoke_home = Path(os.environ.get("PEOPLEOS_HOME", ".")).resolve()
    smoke_home.mkdir(parents=True, exist_ok=True)
    failure_log = smoke_home / "desktop-smoke-failure.log"
    failure_log.unlink(missing_ok=True)

    env = os.environ.copy()
    env["PEOPLEOS_SMOKE_TEST"] = "1"
    env["PEOPLEOS_HOME"] = str(smoke_home)

    try:
        result = subprocess.run(
            [str(exe.resolve())],
            env=env,
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        print("Packaged PeopleOS timed out after 90 seconds.", file=sys.stderr)
        if exc.stdout:
            print(exc.stdout)
        if exc.stderr:
            print(exc.stderr, file=sys.stderr)
        _print_failure_log(smoke_home)
        return 124

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    if result.returncode != 0:
        print(
            f"Packaged PeopleOS exited with code {result.returncode}.",
            file=sys.stderr,
        )
        _print_failure_log(smoke_home)
        return result.returncode or 1

    if failure_log.exists():
        print(
            "Packaged PeopleOS exited successfully but left a failure log.",
            file=sys.stderr,
        )
        _print_failure_log(smoke_home)
        return 1

    print("Packaged PeopleOS smoke test PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
