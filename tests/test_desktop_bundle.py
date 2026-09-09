"""Portable archive contract checks; no desktop or analytics dependencies needed."""

import hashlib
import json
import os
from pathlib import Path
import tarfile
import tempfile
import unittest
import zipfile

from scripts.bundle_desktop import REQUIRED_DOCS, bundle


class DesktopBundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dist = self.root / "dist"
        self.dist.mkdir()
        for name in (*REQUIRED_DOCS, "NOTICE", "licenses/library.txt", "data/templates/example.csv"):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((name + "\n").encode())
        for name in ("PeopleOS", "PeopleOS.exe"):
            (self.dist / name).write_bytes(b"pretend executable\x00\xff")
        self.revision = "a" * 40

    def test_all_platform_archives_preserve_payload_hashes_and_executable_permissions(self):
        for platform in ("linux-x64", "windows-x64", "macos-arm64"):
            with self.subTest(platform=platform):
                archive = bundle(self.root, self.dist, self.root / platform, platform, self.revision)
                prefix = f"PeopleOS-{platform}/"
                if platform == "windows-x64":
                    with zipfile.ZipFile(archive) as source:
                        contents = {name[len(prefix):]: source.read(name) for name in source.namelist()}
                        self.assertEqual(source.getinfo(prefix + "PeopleOS.exe").external_attr >> 16 & 0o777, 0o755)
                else:
                    with tarfile.open(archive) as source:
                        contents = {member.name[len(prefix):]: source.extractfile(member).read() for member in source.getmembers()}
                        self.assertEqual(source.getmember(prefix + "PeopleOS").mode, 0o755)
                provenance = json.loads(contents.pop("BUILD_INFO.json"))
                self.assertEqual(provenance["source_revision"], self.revision)
                self.assertEqual(provenance["platform"], platform)
                self.assertEqual(provenance["files"], {name: hashlib.sha256(value).hexdigest() for name, value in contents.items()})
                for name in (*REQUIRED_DOCS, "NOTICE", "licenses/library.txt", "data/templates/example.csv"):
                    self.assertEqual(contents[name], (self.root / name).read_bytes())
                self.assertEqual((archive.parent / "SHA256SUMS").read_text(), f"{hashlib.sha256(archive.read_bytes()).hexdigest()}  {archive.name}\n")

    def test_archive_bytes_ignore_input_mtimes_and_modes(self):
        for platform in ("linux-x64", "windows-x64", "macos-arm64"):
            with self.subTest(platform=platform):
                first = bundle(self.root, self.dist, self.root / "first", platform, self.revision).read_bytes()
                for path in [*(self.root / name for name in REQUIRED_DOCS), self.dist / "PeopleOS", self.dist / "PeopleOS.exe"]:
                    os.utime(path, (1_700_000_000, 1_700_000_000))
                    path.chmod(0o600)
                second = bundle(self.root, self.dist, self.root / "second", platform, self.revision).read_bytes()
                self.assertEqual(first, second)

    def test_missing_executable_or_legal_document_fails_before_output(self):
        for name in ("PeopleOS", "LICENSE"):
            path = self.dist / name if name == "PeopleOS" else self.root / name
            original = path.read_bytes()
            path.unlink()
            with self.assertRaisesRegex(ValueError, name):
                bundle(self.root, self.dist, self.root / "absent", "linux-x64", self.revision)
            self.assertFalse((self.root / "absent").exists())
            path.write_bytes(original)

    def test_revision_requires_full_commit_hash(self):
        for revision in ("main", "abc123", "", "A" * 40):
            with self.assertRaisesRegex(ValueError, "commit hash"):
                bundle(self.root, self.dist, self.root / "absent", "linux-x64", revision)


if __name__ == "__main__":
    unittest.main()
