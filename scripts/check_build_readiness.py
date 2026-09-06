"""Automated BUILD READY gate for the PeopleOS transition architecture."""

from __future__ import annotations

import json
from pathlib import Path

from validate_system_model import validate as validate_system_model


REQUIRED_PATHS = [
    "model/system.json",
    "src/platform/workspace.py",
    "src/platform/model_lifecycle.py",
    "src/platform/monitoring.py",
    "src/platform/jobs.py",
    "src/platform/health.py",
    "src/agent/orchestrator.py",
    "src/agent/evidence.py",
    "api/security.py",
    "api/authorization.py",
    "api/routes/intelligence.py",
    "api/routes/platform.py",
    "web/app/platform/page.tsx",
]


def check() -> list[str]:
    errors: list[str] = []
    errors.extend(validate_system_model())

    for path in REQUIRED_PATHS:
        if not Path(path).exists():
            errors.append(f"required architecture artifact missing: {path}")

    model_path = Path("model/system.json")
    if model_path.exists():
        payload = json.loads(model_path.read_text(encoding="utf-8"))
        if payload.get("build_readiness") == "NOT BUILD READY":
            errors.append("canonical model declares NOT BUILD READY")
        controls = payload.get("controls", [])
        required_controls = {"ctl-local", "ctl-agent-tools", "ctl-small-groups", "ctl-llm", "ctl-model-activation", "ctl-autonomy"}
        present_controls = {item.get("id") for item in controls}
        missing_controls = required_controls - present_controls
        if missing_controls:
            errors.append(f"missing required controls: {sorted(missing_controls)}")

    package_path = Path("web/package.json")
    if package_path.exists():
        package = json.loads(package_path.read_text(encoding="utf-8"))
        deps = package.get("dependencies", {})
        next_version = str(deps.get("next", ""))
        react_version = str(deps.get("react", ""))
        if not next_version.startswith("16"):
            errors.append(f"frontend must target Next.js 16, found {next_version!r}")
        if not react_version.startswith("19"):
            errors.append(f"frontend must target React 19, found {react_version!r}")

    gitignore = Path(".gitignore").read_text(encoding="utf-8") if Path(".gitignore").exists() else ""
    if ".peopleos/" not in gitignore:
        errors.append(".peopleos/ control-plane state must be excluded from Git")

    readme = Path("README.md").read_text(encoding="utf-8") if Path("README.md").exists() else ""
    if "--host 0.0.0.0" in readme:
        errors.append("README still recommends insecure default API bind 0.0.0.0")

    return errors


if __name__ == "__main__":
    failures = check()
    if failures:
        for failure in failures:
            print(f"NOT READY: {failure}")
        raise SystemExit(1)
    print("BUILD READY WITH ASSUMPTIONS: PeopleOS architecture gates passed")
