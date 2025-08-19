# -*- mode: python ; coding: utf-8 -*-

import os
from PyInstaller.utils.hooks import collect_all, collect_dynamic_libs, collect_data_files

block_cipher = None

# Собираем всё из faster_whisper и ctranslate2
fw_datas, fw_binaries, fw_hidden = collect_all('faster_whisper')
ct2_datas, ct2_binaries, ct2_hidden = collect_all('ctranslate2')

# Подстраховка: динамические библиотеки CTranslate2
ct2_binaries += collect_dynamic_libs('ctranslate2')
ct2_datas    += collect_data_files('ctranslate2')

# Приложим ffmpeg, если есть в third_party/ffmpeg/bin
extra_bins = []
ffmpeg_bin_dir = os.path.join('third_party', 'ffmpeg', 'bin')
if os.path.isdir(ffmpeg_bin_dir):
    for fn in os.listdir(ffmpeg_bin_dir):
        full = os.path.join(ffmpeg_bin_dir, fn)
        if os.path.isfile(full):
            extra_bins.append((full, '.'))

a = Analysis(
    ['stt_cli.py'],
    pathex=[],
    binaries=fw_binaries + ct2_binaries + extra_bins,
    datas=fw_datas + ct2_datas,
    hiddenimports=fw_hidden + ct2_hidden + ['faster_whisper', 'ctranslate2'],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    noarchive=False
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=False,
    name='stt_cli',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    name='stt_cli'
)
