import sqlite3
import threading
from pathlib import Path

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from core.hasher import hash_file
import sqlite_vec


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


def _connect(db_path: str) -> sqlite3.Connection:
    db = sqlite3.connect(str(Path(db_path).expanduser()))
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    db.execute("PRAGMA journal_mode=WAL")
    return db


class FerretEventHandler(FileSystemEventHandler):
    def __init__(self, db_path: str, model_path: str = "~/ferret/models/bge-small-en"):
        super().__init__()
        self.db_path = db_path
        self.model_path = model_path
        self._lock = threading.Lock()

    def _supported(self, path: str) -> bool:
        return Path(path).suffix.lower() in SUPPORTED_EXTENSIONS

    def on_created(self, event):
        if event.is_directory or not self._supported(event.src_path):
            return
        print(f"[watcher] Created: {event.src_path}")
        self._index(event.src_path)

    def on_modified(self, event):
        if event.is_directory or not self._supported(event.src_path):
            return
        path = Path(event.src_path)
        with self._lock:
            db = _connect(self.db_path)
            try:
                row = db.execute(
                    "SELECT id, hash FROM files WHERE path = ?", (str(path.resolve()),)
                ).fetchone()
            finally:
                db.close()

        if row:
            current_hash = hash_file(path)
            if current_hash and current_hash != row[1]:
                print(f"[watcher] Modified (hash changed): {path.name}")
                self._delete_and_reindex(str(path.resolve()), row[0])
            else:
                print(f"[watcher] Modified (hash unchanged, skip): {path.name}")
        else:
            print(f"[watcher] Modified (new file): {path.name}")
            self._index(event.src_path)

    def on_deleted(self, event):
        if event.is_directory or not self._supported(event.src_path):
            return
        path = str(Path(event.src_path).resolve())
        print(f"[watcher] Deleted: {path}")
        with self._lock:
            db = _connect(self.db_path)
            try:
                db.execute(
                    "UPDATE files SET status='orphaned' WHERE path = ?", (path,)
                )
                db.commit()
            except Exception as e:
                print(f"[watcher] Error marking orphaned: {e}")
            finally:
                db.close()

    def on_moved(self, event):
        if event.is_directory:
            return
        src = str(Path(event.src_path).resolve())
        dest = str(Path(event.dest_path).resolve())
        dest_path = Path(dest)

        print(f"[watcher] Moved: {src} → {dest}")

        with self._lock:
            db = _connect(self.db_path)
            try:
                if self._supported(dest):
                    db.execute(
                        "UPDATE files SET path=?, filename=?, status='indexed' WHERE path=?",
                        (dest, dest_path.name, src),
                    )
                else:
                    # Moved to unsupported extension — treat as delete
                    db.execute(
                        "UPDATE files SET status='orphaned' WHERE path=?", (src,)
                    )
                db.commit()
            except Exception as e:
                print(f"[watcher] Error on move: {e}")
            finally:
                db.close()

    def _index(self, path: str):
        from core.indexer import index_file
        with self._lock:
            index_file(path, self.db_path, self.model_path)

    def _delete_and_reindex(self, path: str, file_id: int):
        from core.indexer import index_file, _delete_file_data
        with self._lock:
            db = _connect(self.db_path)
            try:
                _delete_file_data(db, file_id)
                db.commit()
            finally:
                db.close()
            index_file(path, self.db_path, self.model_path)


class FolderWatcher:
    """Watches one or more folders for filesystem changes and keeps the index up to date."""

    def __init__(self, db_path: str, model_path: str = "~/ferret/models/bge-small-en"):
        self.db_path = db_path
        self.model_path = model_path
        self._observer = Observer()
        self._handler = FerretEventHandler(db_path, model_path)
        self._watched: set[str] = set()

    def add_folder(self, folder: str | Path) -> None:
        folder = str(Path(folder).resolve())
        if folder in self._watched:
            return
        self._observer.schedule(self._handler, folder, recursive=True)
        self._watched.add(folder)
        print(f"[watcher] Watching: {folder}")

    def start(self) -> None:
        if not self._observer.is_alive():
            self._observer.start()
            print("[watcher] Observer started")

    def stop(self) -> None:
        if self._observer.is_alive():
            self._observer.stop()
            self._observer.join()
            print("[watcher] Observer stopped")
