# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for Ferret.
# The CI workflow exports the ONNX model to models/ before running this.

import os
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

block_cipher = None

# Include sqlite_vec compiled extension (.so / .pyd)
sqlite_vec_datas = collect_data_files("sqlite_vec")
sqlite_vec_bins  = collect_dynamic_libs("sqlite_vec")

# Explicitly include vec0.so since collect_* may miss it
import sqlite_vec as _sv
_sv_dir = Path(_sv.__file__).parent
_vec0 = _sv_dir / ("vec0.dll" if os.name == "nt" else "vec0.so")
if _vec0.exists():
    sqlite_vec_datas.append((str(_vec0), "sqlite_vec"))

# Bundle the pre-exported ONNX model if present
model_src = Path("models")
model_datas = [("models", "models")] if model_src.is_dir() else []

# Bundle assets if present
asset_datas = [("assets", "assets")] if Path("assets").is_dir() else []

all_datas = sqlite_vec_datas + model_datas + asset_datas

hidden_imports = [
    # Local packages
    "core",
    "core.indexer",
    "core.searcher",
    "core.extractor",
    "core.hasher",
    "core.watcher",
    "ui",
    "ui.searchbar",
    "ui.tray",
    "ui.settings",
    # PyQt6
    "PyQt6.QtCore",
    "PyQt6.QtGui",
    "PyQt6.QtWidgets",
    # sqlite
    "sqlite3",
    "sqlite_vec",
    # document extraction
    "fitz",          # PyMuPDF
    "docx",          # python-docx
    "pytesseract",
    "PIL",
    "PIL.Image",
    # ML / inference
    "onnxruntime",
    "numpy",
    "tokenizers",
    # hotkey / file watching
    "pynput.keyboard",
    "pynput.mouse",
    "watchdog.observers",
    "watchdog.observers.inotify",   # Linux
    "watchdog.observers.winapi",    # Windows
    "watchdog.observers.polling",
    "watchdog.events",
    # misc
    "psutil",
    "huggingface_hub",
]

excludes = [
    "torch", "sentence_transformers", "chromadb", "triton",
    "sklearn", "scipy", "transformers", "cuda",
    "nvidia", "grpc", "opentelemetry",
]

a = Analysis(
    ["main.py"],
    pathex=["."],
    binaries=sqlite_vec_bins,
    datas=all_datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ferret",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,           # no terminal window
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon="assets/ferret.png" if Path("assets/ferret.png").exists() else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="ferret",
)
