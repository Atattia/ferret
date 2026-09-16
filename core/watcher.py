import fnmatch
import threading
from pathlib import Path

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


class FerretEventHandler(FileSystemEventHandler):
    def __init__(self, indexing_service, exclude_patterns=()):
        super().__init__()
        self.indexing_service = indexing_service
        self.exclude_patterns = tuple(exclude_patterns or ())

    def _supported(self, path: str) -> bool:
        candidate = Path(path)
        if candidate.suffix.lower() not in SUPPORTED_EXTENSIONS:
            return False
        return not any(
            fnmatch.fnmatchcase(part, pattern)
            for pattern in self.exclude_patterns
            for part in candidate.parts
        )

    def on_created(self, event):
        if event.is_directory or not self._supported(event.src_path):
            return
        print(f"[watcher] Created: {event.src_path}")
        self.indexing_service.enqueue_index(event.src_path)

    def on_modified(self, event):
        if event.is_directory or not self._supported(event.src_path):
            return
        print(f"[watcher] Modified: {event.src_path}")
        # Hash comparison happens on the indexing worker, so bursts of editor
        # events can coalesce before any extraction or database work begins.
        self.indexing_service.enqueue_index(event.src_path)

    def on_deleted(self, event):
        if event.is_directory or not self._supported(event.src_path):
            return
        path = str(Path(event.src_path).resolve())
        print(f"[watcher] Deleted: {path}")
        self.indexing_service.enqueue_delete(path)

    def on_moved(self, event):
        if event.is_directory:
            return
        src = str(Path(event.src_path).resolve())
        dest = str(Path(event.dest_path).resolve())

        print(f"[watcher] Moved: {src} → {dest}")

        source_supported = self._supported(src)
        destination_supported = self._supported(dest)
        if source_supported and destination_supported:
            self.indexing_service.enqueue_move(src, dest)
        elif source_supported:
            self.indexing_service.enqueue_delete(src)
        elif destination_supported:
            self.indexing_service.enqueue_index(dest)


class FolderWatcher:
    """Watches one or more folders for filesystem changes and keeps the index up to date."""

    def __init__(
        self,
        db_path: str,
        model_path: str = "~/ferret/models/bge-small-en",
        *,
        indexing_service=None,
        exclude_patterns=(),
    ):
        self._owns_indexing_service = indexing_service is None
        if indexing_service is None:
            from core.indexing_service import IndexingService

            indexing_service = IndexingService(db_path, model_path)
        self.db_path = db_path
        self.model_path = model_path
        self.indexing_service = indexing_service
        self._observer = Observer()
        self._handler = FerretEventHandler(indexing_service, exclude_patterns)
        self._watched: set[str] = set()
        self._watches: dict[str, object] = {}
        self._lock = threading.RLock()
        self._running = False
        self._started_once = False

    @staticmethod
    def _normalise_folder(folder: str | Path) -> str | None:
        """Return a canonical directory path, or ``None`` if it is not a folder.

        Settings can contain paths that have since been removed.  Skipping those
        paths here keeps a stale setting from preventing the rest of the watcher
        from starting.
        """
        try:
            path = Path(folder).expanduser().resolve()
        except (TypeError, ValueError, OSError):
            return None
        return str(path) if path.is_dir() else None

    def _schedule_locked(self, folder: str) -> bool:
        if folder in self._watches:
            return True
        # A directory may disappear between configuration and scheduling.
        if not Path(folder).is_dir():
            print(f"[watcher] Folder unavailable, skipping: {folder}")
            return False
        try:
            watch = self._observer.schedule(self._handler, folder, recursive=True)
        except (OSError, RuntimeError) as exc:
            print(f"[watcher] Could not watch {folder}: {exc}")
            return False
        self._watches[folder] = watch
        print(f"[watcher] Watching: {folder}")
        return True

    def add_folder(self, folder: str | Path) -> bool:
        """Add a valid directory to the watch set.

        Returns whether the folder is configured for watching.  Invalid or
        missing folders are ignored rather than raising during application
        startup or settings changes.
        """
        folder = self._normalise_folder(folder)
        if folder is None:
            return False
        with self._lock:
            if folder in self._watched:
                return True
            self._watched.add(folder)
            self._schedule_locked(folder)
            return True

    def remove_folder(self, folder: str | Path) -> bool:
        """Stop watching a folder and unschedule its watchdog watch."""
        try:
            folder = str(Path(folder).expanduser().resolve())
        except (TypeError, ValueError, OSError):
            return False
        with self._lock:
            was_watched = folder in self._watched
            self._watched.discard(folder)
            watch = self._watches.pop(folder, None)
            if watch is not None:
                try:
                    self._observer.unschedule(watch)
                except (OSError, RuntimeError, ValueError) as exc:
                    # Unscheduling a watch that watchdog has already removed is
                    # harmless; retain the desired set and continue.
                    print(f"[watcher] Could not unschedule {folder}: {exc}")
            return was_watched

    def reconfigure(self, folders) -> set[str]:
        """Replace the configured folder set without disrupting other watches.

        The returned set contains only existing directories that are now being
        watched.  Existing observer threads remain running while paths are
        added/removed.
        """
        desired = set()
        for folder in folders or []:
            normalised = self._normalise_folder(folder)
            if normalised is not None:
                desired.add(normalised)

        with self._lock:
            for folder in self._watched - desired:
                self.remove_folder(folder)
            for folder in desired - self._watched:
                self._watched.add(folder)
                self._schedule_locked(folder)
            return set(self._watched)

    @property
    def watched_folders(self) -> set[str]:
        with self._lock:
            return set(self._watched)

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            if not self._watched:
                print("[watcher] No valid folders configured; observer not started")
                return

            # watchdog observers cannot be started again after stop(). Create a
            # fresh observer and restore the configured watches on restart.
            if self._started_once:
                self._observer = Observer()
                self._watches = {}
            for folder in self._watched:
                self._schedule_locked(folder)
            if not self._watches:
                print("[watcher] No valid folders available; observer not started")
                return
            try:
                self._observer.start()
            except (OSError, RuntimeError) as exc:
                print(f"[watcher] Could not start observer: {exc}")
                return
            self._running = True
            self._started_once = True
            if self._owns_indexing_service:
                self.indexing_service.start()
            print("[watcher] Observer started")

    def stop(self) -> None:
        with self._lock:
            if not self._running and not self._observer.is_alive():
                return
            try:
                self._observer.stop()
                self._observer.join()
            finally:
                self._running = False
            if self._owns_indexing_service:
                self.indexing_service.stop(drain=False, timeout=2)
            print("[watcher] Observer stopped")
