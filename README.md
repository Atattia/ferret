# Ferret

A local semantic search tool for Linux. Ferret runs as a system tray app, watches your folders for file changes, indexes documents using a local AI embedding model, and lets you search your files by meaning — not just keywords — with a hotkey-triggered search bar.

Everything runs locally. No data leaves your machine.

## Features

- **Semantic search** — finds documents by meaning, not exact words
- **Hotkey-triggered** — press `Ctrl+Space` anywhere to open the search bar
- **System tray** — runs quietly in the background
- **Supported formats** — `.pdf`, `.docx`, `.txt`, `.md`
- **OCR support** — extracts text from scanned PDFs via pytesseract or PaddleOCR
- **Local model** — uses BGE-small-en (384-dim) via ONNX; no internet required
- **SQLite storage** — vectors stored locally via `sqlite-vec`
- **Configurable** — choose folders, indexing speed profile, and OCR engine from the settings dialog

## Installation

### Ubuntu / Debian — download and double-click

1. Go to the [Releases](../../releases) page and download `ferret_<version>_amd64.deb`
2. Double-click the file — Ubuntu Software Center will open and install it
3. Launch **Ferret** from the application menu, or run `ferret` in a terminal

The AI model (~120 MB) is bundled inside the package — no internet connection needed after installation.

> **OCR for scanned PDFs** requires Tesseract, which is installed automatically as a dependency.

### Windows — download and run the installer

1. Go to the [Releases](../../releases) page and download `ferret_<version>_windows_setup.exe`
2. Run the installer and follow the prompts
3. Ferret appears in the Start menu

> **OCR for scanned PDFs** on Windows requires [Tesseract for Windows](https://github.com/UB-Mannheim/tesseract/wiki) installed separately.

---

## Development Setup

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/YOUR_USERNAME/ferret.git
cd ferret
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-core.txt
```

### 2. Download the embedding model

```bash
pip install "optimum[exporters]"
optimum-cli export onnx --model BAAI/bge-small-en --task feature-extraction ~/ferret/models/bge-small-en/
```

### 3. Run

```bash
source venv/bin/activate
python main.py
```

The app starts in the system tray. The database is auto-created at `~/ferret/ferret.db` on first run.

## Usage

1. **Open settings** — right-click the tray icon and choose Settings
2. **Add folders** — select folders you want Ferret to index
3. **Trigger indexing** — right-click the tray icon and choose Re-index
4. **Search** — press `Ctrl+Space` to open the search bar, type a query, and press Enter

## Configuration

Settings are saved to `config/settings.json`. You can edit it manually or use the Settings dialog.

| Key | Default | Description |
|---|---|---|
| `indexed_folders` | `[]` | Folders to index |
| `exclude_patterns` | `["node_modules", ".git", "venv", "__pycache__"]` | Patterns to skip |
| `ocr_engine` | `"pytesseract"` | OCR backend (`pytesseract` or `PaddleOCR`) |
| `indexing_workers` | `4` | Parallel indexing workers |
| `model_path` | `~/ferret/models/bge-small-en` | Path to the ONNX model directory |
| `db_path` | `~/ferret/ferret.db` | Path to the SQLite database |

**Indexing speed profiles** (selectable in Settings):

| Profile | Workers | Notes |
|---|---|---|
| Safe | 2 | Low resource usage |
| Balanced | 4 | Default, good for most laptops |
| Fast | 8 | For powerful desktops |
| Maximum | All cores | Uses every available CPU core |

## Project Structure

```
ferret/
├── main.py               # Entry point
├── core/
│   ├── extractor.py      # Text extraction (PDF, DOCX, TXT, MD)
│   ├── indexer.py        # Chunking, embedding, DB storage
│   ├── searcher.py       # Vector search with orphan detection
│   ├── watcher.py        # Filesystem change monitoring
│   └── hasher.py         # SHA256 file fingerprinting
├── ui/
│   ├── searchbar.py      # Frameless search bar (PyQt6)
│   ├── tray.py           # System tray icon and menu
│   └── settings.py       # Settings dialog
└── config/
    └── settings.json     # User config (auto-created, gitignored)
```

## License

MIT
