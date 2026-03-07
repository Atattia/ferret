import json
import os
import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication

from core.indexer import init_db, rebuild_fts
from ui.searchbar import SearchBar
from ui.tray import FerretTray
from ui.settings import SettingsWindow


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
        bundled = Path(sys._MEIPASS) / "models" / "bge-small-en"
        if (bundled / "onnx" / "model.onnx").exists():
            return str(bundled)
    return None


CONFIG_PATH = _get_config_path()


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
    config = load_config()
    db_path = str(Path(config["db_path"]).expanduser())
    model_path = config.get("model_path", "~/ferret/models/bge-small-en")

    # Fall back to bundled model if the configured path doesn't have the files
    resolved = Path(model_path).expanduser()
    if not (resolved / "onnx" / "model.onnx").exists():
        bundled = _get_bundled_model_path()
        if bundled:
            print(f"[main] Using bundled model: {bundled}")
            model_path = bundled

    init_db(db_path)
    rebuild_fts(db_path)

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    search_bar = SearchBar(db_path=db_path, model_path=model_path)

    def open_settings():
        win = SettingsWindow(config, on_save=lambda new: save_config(config, new), parent=None)
        win.exec()

    def _run_index(force: bool = False):
        from core.indexer import index_folder, reset_file_hashes
        import threading
        folders = config.get("indexed_folders", [])
        workers = config.get("indexing_workers", 4)
        if not folders:
            print("[main] No folders configured for indexing")
            return
        def _run():
            if force:
                reset_file_hashes(db_path)
            for folder in folders:
                index_folder(folder, db_path, workers=workers, model_path=model_path)
        threading.Thread(target=_run, daemon=True).start()

    tray = FerretTray(
        app,
        on_settings=open_settings,
        on_reindex=lambda: _run_index(force=False),
        on_force_reindex=lambda: _run_index(force=True),
        on_quit=app.quit,
    )

    def toggle_search():
        if search_bar.isVisible():
            search_bar.hide()
        else:
            search_bar.show_and_focus()

    _hotkey = setup_hotkey(toggle_search)

    print("Ferret is running")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
