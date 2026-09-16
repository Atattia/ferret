# Ferret

A local semantic search tool for Linux. Ferret runs as a system tray app, watches your folders for file changes, indexes documents using a local AI embedding model, and lets you search your files by meaning — not just keywords — with a hotkey-triggered search bar.

Everything runs locally. No data leaves your machine.

For the new bilingual model pipeline, Arabic OCR, safe upgrades, and benchmark
commands, see [the search upgrade guide](docs/SEARCH_UPGRADE.md). Existing English
indexes must be rebuilt into a separate index before selecting a multilingual model.

## Features

- **Semantic search** — finds documents by meaning, not exact words
- **Hybrid ranking** — combines filename, keyword (BM25), document-type, and semantic results using reciprocal rank fusion
- **Scoped search** — use `type:pdf,md` and `in:~/Documents` to narrow every retrieval route
- **Match evidence** — previews show the matching passage, retrieval routes, and page or line
- **Headless CLI** — search, export JSON results, and inspect index health without starting Qt
- **Hotkey-triggered** — press `Ctrl+Space` anywhere to open the search bar
- **System tray** — runs quietly in the background
- **Supported formats** — `.pdf`, `.docx`, `.txt`, `.md`
- **OCR support** — extracts English and Arabic scanned PDFs via Tesseract
- **Local models** — manifest-based ONNX adapters for multilingual E5, BGE-M3, Qwen3 embeddings, and BGE multilingual reranking; legacy BGE-small remains readable
- **SQLite storage** — vectors stored locally via `sqlite-vec`
- **Configurable** — choose folders, indexing speed profile, and OCR languages from the settings dialog

## Installation

### Ubuntu / Debian — download and double-click

1. Go to the [Releases](../../releases) page and download `ferret_<version>_amd64.deb`
2. Double-click the file — Ubuntu Software Center will open and install it
3. Launch **Ferret** from the application menu, or run `ferret` in a terminal

Released packages bundle their selected model. The new multilingual pipeline needs
the explicit model download and rebuild described in the upgrade guide; existing
release binaries have not been rebuilt by this source change.

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
pip install -r requirements.txt
```

For runtime-only installation, use `requirements-core.txt`. Unit tests use the
standard-library `unittest` runner:

```bash
QT_QPA_PLATFORM=offscreen python -m unittest discover -s tests -p '*unit.py'
```

Model weights, personal settings, indexes, caches, generated benchmark reports,
and local coding-agent guidance are excluded from Git. Benchmark code and the
[results summary](benchmarks/RESULTS.md) are kept; raw reports remain local.

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

Use ↑/↓ to select a result, Enter to open it, `Ctrl+Shift+C` to copy its path,
and Escape to dismiss. Results open with your platform's default application.
The search bar runs one search at a time and keeps only the latest pending query.

### Search examples

```text
customer retention strategy
annual revenue type:pdf
deployment notes type:md,txt in:~/Documents
budget in:"/home/me/Work Documents"
type:pdf
```

`type:` accepts comma-separated extensions; `in:` includes descendants of one
folder (the last `in:` wins). Filters can be used without text to browse indexed
files. Quotes group paths containing spaces; they do not request exact phrase
search. Unknown operators remain ordinary query text.

### Command line

Run from the repository with its virtual environment activated:

```bash
python cli.py 'annual revenue type:pdf'
python cli.py 'deployment notes' --mode keyword --json
python cli.py 'customer retention' --mode semantic -k 10
python cli.py --db ~/ferret/ferret.db --stats
```

Use `--db` and `--model` if your app uses custom locations. CLI defaults are
`~/ferret/ferret.db` and `~/ferret/models/bge-small-en`; it does not load the GUI
settings. Keyword mode requires no model inference. JSON includes ranked results,
match evidence, and elapsed search time; diagnostics go to stderr.

### Search architecture and limits

Inspired by [Alibaba zvec](https://github.com/alibaba/zvec)'s embedded hybrid
retrieval and structured filters, Ferret keeps its existing SQLite + sqlite-vec
storage. No new server, vector database dependency, or data migration is needed.
Scoped semantic queries compute exact cosine distances for eligible files before
selecting candidates, so unrelated global neighbors cannot hide scoped matches.
This favors recall; large scoped collections still require a linear vector scan.
Ferret does not currently implement an approximate nearest-neighbor index or
claim zvec's performance characteristics.

Queries preserve natural-language relationships such as “without” for embeddings.
Search connections are read-only, and filesystem reconciliation runs separately
from retrieval. The old universal semantic distance cutoff has been removed;
optional relevance rejection uses model-specific development calibration. Ranking quality
depends on your documents and local embedding model. Result scores are relative
fusion ranks, not confidence probabilities.

### Tests

```bash
QT_QPA_PLATFORM=offscreen python -m unittest discover -s tests -p '*unit.py'
```

These tests cover extraction, chunking, embedding conventions, incremental indexing,
watching, reconciliation, hybrid ranking, scoped vector recall, CLI output, and
desktop search concurrency. Vector retrieval tests use deterministic embeddings;
they do not measure relevance on a real document collection.

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
│   ├── searcher.py       # Read-only hybrid retrieval and rank fusion
│   ├── query.py          # Shared type/folder filter grammar
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
