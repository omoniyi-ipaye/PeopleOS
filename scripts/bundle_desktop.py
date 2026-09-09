"""Wrap an already smoke-tested desktop executable in a stable release archive.

Reproducibility applies to archive metadata for identical input bytes; PyInstaller
executables and dependency resolution are not claimed to be reproducible builds.
No release is published and no code signature is added by this script.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import tarfile
import zipfile

PLATFORMS = ("linux-x64", "windows-x64", "macos-arm64")
REQUIRED_DOCS = ("LICENSE", "README.md", "docs/architecture/LOCAL_DISTRIBUTION.md",
                 "docs/PUBLIC_BETA_GUIDE.md", "SECURITY.md", "CONTRIBUTING.md",
                 "docs/releases/PUBLIC_BETA_CHECKLIST.md", "sample_hr_data.csv")


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def bundle(root: Path, dist: Path, output: Path, platform: str, revision: str) -> Path:
    if platform not in PLATFORMS:
        raise ValueError(f"Unsupported platform: {platform}")
    if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", revision):
        raise ValueError("Source revision must be a full lowercase Git commit hash")
    executable = "PeopleOS.exe" if platform == "windows-x64" else "PeopleOS"
    files = {executable: dist / executable}
    files.update({name: root / name for name in REQUIRED_DOCS})
    for pattern in ("NOTICE*", "THIRD_PARTY*", "licenses/**/*", "docs/beta/**/*", "docs/releases/**/*", "data/templates/**/*"):
        for path in sorted(root.glob(pattern)):
            if path.is_file():
                files[path.relative_to(root).as_posix()] = path
    for name, path in files.items():
        if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing, empty, or symlinked bundle input: {name}")
    # Keep a hash of every payload, including the legal notices and user docs.
    provenance = (json.dumps({
        "schema_version": 1,
        "product": "PeopleOS",
        "platform": platform,
        "source_revision": revision,
        "files": {name: sha256(path) for name, path in sorted(files.items())},
    }, indent=2, sort_keys=True) + "\n").encode("utf-8")
    basename = f"PeopleOS-{platform}"
    extension = ".zip" if platform == "windows-x64" else ".tar.gz"
    output.mkdir(parents=True, exist_ok=True)
    archive = output / (basename + extension)
    entries = sorted([*files, "BUILD_INFO.json"])

    def open_entry(name: str):
        return io.BytesIO(provenance) if name == "BUILD_INFO.json" else files[name].open("rb")

    if extension == ".zip":
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as target:
            for name in entries:
                info = zipfile.ZipInfo(f"{basename}/{name}", date_time=(1980, 1, 1, 0, 0, 0))
                info.create_system = 3
                info.external_attr = (0o100755 if name == executable else 0o100644) << 16
                info.compress_type = zipfile.ZIP_DEFLATED
                with open_entry(name) as source, target.open(info, "w", force_zip64=True) as dest:
                    shutil.copyfileobj(source, dest)
    else:
        with archive.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as target:
                for name in entries:
                    info = tarfile.TarInfo(f"{basename}/{name}")
                    info.mode = 0o755 if name == executable else 0o644
                    info.size = len(provenance) if name == "BUILD_INFO.json" else files[name].stat().st_size
                    # TarInfo defaults pin uid/gid, names, and mtime to neutral values.
                    with open_entry(name) as source:
                        target.addfile(info, source)
    (output / "SHA256SUMS").write_text(f"{sha256(archive)}  {archive.name}\n", encoding="utf-8", newline="\n")
    return archive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dist", type=Path, default=Path("dist"))
    parser.add_argument("--output", type=Path, default=Path("release"))
    parser.add_argument("--platform", choices=PLATFORMS, required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    print(bundle(args.root, args.dist, args.output, args.platform, args.revision))


if __name__ == "__main__":
    main()
