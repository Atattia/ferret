# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Ferret is a local semantic search tool for Linux. It runs as a system tray app, watches configured folders for file changes, indexes supported documents (`.pdf`, `.docx`, `.txt`, `.md`) using a local ONNX embedding model, stores vectors in SQLite via `sqlite-vec`, and provides a hotkey-triggered (Ctrl+Space) frameless search bar.

## Running

```bash
source venv/bin/activate
python main.py
```

There are no build steps, linters, or test suites configured. The app logs to stdout with `[module]` prefixes.

## External Resources (user-provided, not in repo)

- **DB**: `~/ferret/ferret.db` — auto-created by `init_db` on first run
- **Model**: `~/ferret/models/bge-small-en/onnx/model.onnx` — BGE-small-en ONNX model (384-dim)
- **Tokenizer**: `~/ferret/models/bge-small-en/tokenizer.json`
- **Config**: `config/settings.json` — edited via the Settings dialog; falls back to hardcoded defaults if missing

## Architecture

### Data flow

```
Files on disk
  → core/extractor.py   (text extraction: pymupdf + OCR fallback for PDFs, python-docx for .docx, plain read for .txt/.md)
  → core/indexer.py     (chunking 500-word windows w/ 50-word overlap, ONNX embedding, sqlite-vec storage)
  → ~/ferret/ferret.db  (files + chunks tables, vec_chunks virtual table)
```

### Search flow

```
Query string
  → core/searcher.py    (embed query → vec_chunks KNN → JOIN chunks/files → orphan detection → top-k results)
  → ui/searchbar.py     (SearchWorker QThread, 350ms debounce, generation counter to discard stale results)
```

### Startup (main.py)

1. Load `config/settings.json`
2. `init_db` — creates DB and schema if not present
3. Launch PyQt6 app with `setQuitOnLastWindowClosed(False)` (tray keeps it alive)
4. Create `SearchBar`, `FerretTray`, `SettingsWindow` (modal dialog)
5. Register `GlobalHotKeys` (pynput) in a daemon thread for Ctrl+Space

### Filesystem watching

`core/watcher.py` — `FolderWatcher` wraps watchdog `Observer` + `FerretEventHandler`. Handles create/modify/delete/move events. Modify events compare SHA256 hashes before re-indexing to avoid redundant work. Not wired into `main.py` yet (watching must be explicitly started).

## Key Implementation Details

- **sqlite-vec loading**: `sqlite_vec.load(db)` after `db.enable_load_extension(True)`, then disable again. Use `sqlite_vec.__version__` (not `.version()`).
- **Vector serialization**: `struct.pack(f"{len(v)}f", *v.tolist())` — little-endian float32 bytes required by sqlite-vec.
- **ONNX inputs**: `input_ids`, `attention_mask`, `token_type_ids` (zeros). Embeddings use mean pooling over token dim with attention mask, then L2-normalized → 384-dim float32.
- **ONNX session cache**: `_session` is a module-level global in `indexer.py`, cached per-process to avoid reloading across chunks in multiprocessing pool workers.
- **KNN query syntax**: `WHERE vc.embedding MATCH ? AND k = ?` with the serialized query vector and k value as positional parameters.
- **Orphan detection**: When a search result's file path no longer exists, `searcher.py` tries to relocate by SHA256 hash before marking `status='orphaned'` in the DB.
- **Search deduplication**: `searcher.py` fetches `top_k * 3` from vec_chunks, then deduplicates by file path, stopping once `top_k` unique files are found.
- **Search concurrency**: `SearchBar` uses a `_generation` counter; `SearchWorker` threads that complete after a newer search was started are silently dropped.
- **Tray icon**: Falls back to a programmatically drawn purple circle if `assets/ferret.png` is not present.