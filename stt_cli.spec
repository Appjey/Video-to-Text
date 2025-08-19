# -*- mode: python ; coding: utf-8 -*-

import os
from PyInstaller.utils.hooks import collect_all, collect_dynamic_libs, collect_data_files

block_cipher = None

# Подтянуть ВСЁ из faster_whisper и ctranslate2 (включая .dll/.pyd)
fw_datas, fw_bins, fw_hidden = collect_all('faster_whisper')
ct2_datas, ct2_bins, ct2_hidden = collect_all('ctranslate2')

# Подстраховка: динамические библиотеки CTranslate2
ct2_bins  += collect_dynamic_libs('ctranslate2')
ct2_datas += collect_data_files('ctranslate2')

# Приложим ffmpeg, если лежит в third_party/ffmpeg/bin
extra_bins = []
ffmpeg_dir = os.path.join('third_party', 'ffmpeg', 'bin')
if os.path.isdir(ffmpeg_dir):
    for fn in os.listdir(ffmpeg_dir):
        full = os.path.join(ffmpeg_dir, fn)
        if os.path.isfile(full):
            # положим рядом с exe (корень распаковки onefile)
            extra_bins.append((full, '.'))

a = Analysis(
    ['stt_cli.py'],
    pathex=[],
    binaries=fw_bins + ct2_bins + extra_bins,
    datas=fw_datas + ct2_datas,
    hiddenimports=fw_hidden + ct2_hidden + ['faster_whisper', 'ctranslate2'],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    noarchive=False
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# ONEFILE: передаём a.binaries/a.zipfiles/a.datas прямо в EXE и НЕ используем COLLECT
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    name='stt_cli',
    debug=False,
    strip=False,
    upx=True,
    console=True
)

# Никакого COLLECT: это был бы onedir
