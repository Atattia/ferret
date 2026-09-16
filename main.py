import json
import os
import sys
import threading
from pathlib import Path

from PyQt6.QtCore import QObject, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import QApplication

from core.indexer import init_db, rebuild_fts
from core.indexing_service import IndexingService
from core.reconciler import queue_reconciliation_actions, reconcile_filesystem
from core.watcher import FolderWatcher
from ui.searchbar import SearchBar
from ui.tray import FerretTray
from ui.settings import SettingsWindow
from core.maintenance import config_lock
from core.extractor import configure_ocr
from core.models import model_spec, configured_reranker


def _get_config_path() -> Path:
    """Return config path: user data dir when packaged, project dir in dev."""
    if getattr(sys, "frozen", False):
        if sys.platform == "win32":
            base = Path(os.environ.get("APPDATA", Path.home()))
        else:
            base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
        config_dir = base / "ferret"
        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir / "settings.json"
    return Path(__file__).parent / "config" / "settings.json"


def _get_bundled_model_path() -> str | None:
    """Return the model path bundled inside a PyInstaller package, if present."""
    if getattr(sys, "frozen", False):
        for name in ("qwen3-embedding-0.6b", "bge-m3", "multilingual-e5-small", "bge-small-en"):
            bundled = Path(sys._MEIPASS) / "models" / name
            if (bundled / model_spec(bundled).onnx_file).exists():
                return str(bundled)
    return None


CONFIG_PATH = _get_config_path()

# These locations contain generated/dependency text that overwhelms useful
# personal documents when broad roots such as ~/Documents are indexed.
BUILTIN_EXCLUDE_PATTERNS = (
    ".uv-cache",
    ".venv",
    "site-packages",
    ".build",
    "__pycache__",
    "node_modules",
)


class SearchToggleDispatcher(QObject):
    """Marshal global-hotkey callbacks onto Qt's GUI thread."""

    toggle_requested = pyqtSignal()

    def __init__(self, search_bar):
        super().__init__()
        self._search_bar = search_bar
        self.toggle_requested.connect(
            self._toggle_search,
            Qt.ConnectionType.QueuedConnection,
        )

    @pyqtSlot()
    def _toggle_search(self):
        if self._search_bar.isVisible():
            self._search_bar.hide()
        else:
            self._search_bar.show_and_focus()


def load_config() -> dict:
    try:
        with open(CONFIG_PATH) as f:
            return json.load(f)
    except Exception as e:
        print(f"[main] Could not load config: {e}, using defaults")
        return {
            "indexed_folders": [],
            "exclude_patterns": ["node_modules", ".git", "venv", "__pycache__"],
            "ocr_engine": "pytesseract",
            "indexing_workers": 4,
            "model_path": "~/ferret/models/bge-small-en",
            "db_path": "~/ferret/ferret.db",
            "hotkey": "<ctrl>+<space>",
        }


def save_config(live_config: dict, new_values: dict) -> None:
    live_config.update(new_values)
    try:
        with open(CONFIG_PATH, "w") as f:
            json.dump(live_config, f, indent=2)
        print("[main] Config saved")
    except Exception as e:
        print(f"[main] Could not save config: {e}")


def setup_hotkey(callback):
    """Register global hotkey using pynput in a background thread."""
    try:
        from pynput import keyboard

        def on_activate():
            callback()

        hotkey = keyboard.GlobalHotKeys({"<ctrl>+<space>": on_activate})
        hotkey.daemon = True
        hotkey.start()
        print("[main] Global hotkey registered: Ctrl+Space")
        return hotkey
    except Exception as e:
        print(f"[main] Hotkey registration failed: {e}")
        return None


def main():
    try:
        with config_lock(CONFIG_PATH):
            _main()
    except Exception as e:
        # Show a visible error since console=False hides everything
        try:
            app = QApplication.instance() or QApplication(sys.argv)
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(None, "Ferret – Startup Error", str(e))
        except Exception:
            pass
        sys.exit(1)


def _main():
    config = load_config()
    db_path = str(Path(config["db_path"]).expanduser())
    model_path = config.get("model_path", "~/ferret/models/bge-small-en")

    # Ensure data directory exists (fresh install won't have ~/ferret/)
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    # Fall back to bundled model if the configured path doesn't have the files
    resolved = Path(model_path).expanduser()
    if not (resolved / model_spec(resolved).onnx_file).exists():
        bundled = _get_bundled_model_path()
        if bundled:
            print(f"[main] Using bundled model: {bundled}")
            model_path = bundled

    configure_ocr(config)
    init_db(db_path, model_path)
    rebuild_fts(db_path)

    exclude_patterns = list(dict.fromkeys(
        [*config.get("exclude_patterns", []), *BUILTIN_EXCLUDE_PATTERNS]
    ))
    indexing_service = IndexingService(
        db_path,
        model_path,
        exclude_patterns=exclude_patterns,
    )
    indexing_service.start()

    # Keep the filesystem index fresh while the tray application is running.
    # FolderWatcher ignores missing configured directories, so a stale setting
    # cannot prevent startup.
    folder_watcher = FolderWatcher(
        db_path=db_path,
        model_path=model_path,
        indexing_service=indexing_service,
        exclude_patterns=exclude_patterns,
    )
    folder_watcher.reconfigure(config.get("indexed_folders", []))
    folder_watcher.start()

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    search_bar = SearchBar(db_path=db_path, model_path=model_path)
    search_bar.reranker_path = configured_reranker(config)
    search_bar.rerank_minimum = config.get("rerank_minimum")
    search_bar.calibration_path = config.get("calibration_path")
    search_bar.indexing_status = lambda: indexing_service.status

    def _reconcile_and_queue(folders):
        """Detect changes missed while Ferret was closed, then queue them."""
        try:
            actions = reconcile_filesystem(
                folders,
                db_path,
                exclude_patterns=exclude_patterns,
            )
            queue_reconciliation_actions(actions, indexing_service)
            print(f"[main] Reconciliation queued {len(actions)} change(s)")
        except Exception as e:
            print(f"[main] Reconciliation failed: {e}")

    def _start_reconciliation(folders=None):
        selected = list(
            folders if folders is not None else config.get("indexed_folders", [])
        )
        threading.Thread(
            target=_reconcile_and_queue,
            args=(selected,),
            name="ferret-reconcile",
            daemon=True,
        ).start()

    def _save_settings(new_config):
        save_config(config, new_config)
        configure_ocr(config)
        search_bar.reranker_path = configured_reranker(config)
        folder_watcher.reconfigure(config.get("indexed_folders", []))
        # The watcher deliberately stays stopped when the app launches with no
        # folders. Start it here when the user configures the first folder.
        folder_watcher.start()
        _start_reconciliation()

    def open_settings():
        win = SettingsWindow(config, on_save=_save_settings, parent=None)
        win.exec()

    def _run_index(force: bool = False):
        from core.indexer import reset_file_hashes
        folders = config.get("indexed_folders", [])
        if not folders:
            print("[main] No folders configured for indexing")
            return
        def _run():
            if force:
                indexing_service.wait_for_idle()
                reset_file_hashes(db_path)
            _reconcile_and_queue(folders)
        threading.Thread(
            target=_run, name="ferret-manual-reconcile", daemon=True
        ).start()

    # Reconciliation covers normal offline changes and embedding migrations:
    # stale records are emitted as changed even when their file hash matches.
    _start_reconciliation()

    tray = FerretTray(
        app,
        on_settings=open_settings,
        on_reindex=lambda: _run_index(force=False),
        on_force_reindex=lambda: _run_index(force=True),
        on_quit=app.quit,
    )

    # pynput invokes callbacks on its own thread. Emitting this signal is
    # thread-safe; the queued slot above always executes on Qt's main thread.
    _ui_dispatcher = SearchToggleDispatcher(search_bar)
    _hotkey = setup_hotkey(_ui_dispatcher.toggle_requested.emit)

    # aboutToQuit also covers OS/window-manager shutdown and keeps the
    # watchdog thread from surviving the Qt event loop.
    def _shutdown():
        folder_watcher.stop()
        indexing_service.stop(drain=False, timeout=2)

    app.aboutToQuit.connect(_shutdown)

    print("Ferret is running")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
