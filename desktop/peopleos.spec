# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller specification for the self-contained PeopleOS local product."""

from PyInstaller.utils.hooks import collect_submodules

hiddenimports = collect_submodules('api') + collect_submodules('src') + collect_submodules('desktop')

datas = [
    ('config.yaml', '.'),
    ('sample_hr_data.csv', '.'),
    ('data/templates', 'data/templates'),
    ('web/out', 'peopleos_ui'),
]

a = Analysis(
    ['desktop/launcher.py'],
    pathex=['.'],
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
