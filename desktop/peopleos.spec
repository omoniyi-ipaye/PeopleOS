# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller specification for the self-contained PeopleOS local product."""

from pathlib import Path
from PyInstaller.utils.hooks import collect_submodules

ROOT = Path(SPECPATH).resolve().parent

hiddenimports = (
    collect_submodules('api')
    + collect_submodules('src')
    + collect_submodules('desktop')
    + collect_submodules('backports')
)

datas = [
    (str(ROOT / 'config.yaml'), '.'),
    (str(ROOT / 'sample_hr_data.csv'), '.'),
    (str(ROOT / 'data' / 'templates'), 'data/templates'),
    (str(ROOT / 'web' / 'out'), 'peopleos_ui'),
]

a = Analysis(
    [str(ROOT / 'desktop' / 'launcher.py')],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='PeopleOS',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
)
