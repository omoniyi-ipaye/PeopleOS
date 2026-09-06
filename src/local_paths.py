"""OS-native local storage layout for the PeopleOS desktop product.

All mutable user data lives outside the application bundle so installing or
upgrading PeopleOS cannot overwrite workforce data, investigations, or model
artifacts. PEOPLEOS_HOME remains an explicit escape hatch for development,
portable installs, and tests.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PeopleOSPaths:
    root: Path
    database: Path
    registry: Path
    datasets: Path
    models: Path
    investigations: Path
    exports: Path
    logs: Path
    config: Path
    backups: Path

    def ensure(self) -> "PeopleOSPaths":
        for path in (
            self.root,
            self.database.parent,
            self.registry.parent,
            self.datasets,
            self.models,
            self.investigations,
            self.exports,
            self.logs,
            self.config,
            self.backups,
        ):
            path.mkdir(parents=True, exist_ok=True)
        return self


def _default_root() -> Path:
    override = os.getenv("PEOPLEOS_HOME")
    if override:
        return Path(override).expanduser().resolve()

    home = Path.home()
    if sys.platform == "darwin":
        return home / "Library" / "Application Support" / "PeopleOS"
    if os.name == "nt":
        base = Path(os.getenv("LOCALAPPDATA") or os.getenv("APPDATA") or home)
        return base / "PeopleOS"

    xdg = os.getenv("XDG_DATA_HOME")
    base = Path(xdg).expanduser() if xdg else home / ".local" / "share"
    return base / "peopleos"


def get_peopleos_paths(*, ensure: bool = True) -> PeopleOSPaths:
    root = _default_root()
    paths = PeopleOSPaths(
        root=root,
        database=root / "data" / "peopleos.db",
        registry=root / "control-plane" / "workspaces.json",
        datasets=root / "datasets",
        models=root / "models",
        investigations=root / "investigations",
        exports=root / "exports",
        logs=root / "logs",
        config=root / "config",
        backups=root / "backups",
    )
    return paths.ensure() if ensure else paths
