"""Single-writer coordination for Ferret indexing work.

``IndexingService`` is intentionally independent of Qt and watchdog.  UI,
startup, and filesystem-watcher code can all submit work without performing a
database write on their own threads.  The expensive/default indexer is loaded
only when a job runs, which also makes the service straightforward to unit
test with small injected callables.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import sqlite3
import threading
import time
from typing import Callable, Iterable


SUPPORTED_EXTENSIONS = frozenset({".pdf", ".docx", ".txt", ".md"})


class JobKind(str, Enum):
    INDEX = "index"
    DELETE = "delete"
    MOVE = "move"
    SCAN = "scan"


@dataclass(frozen=True)
class IndexJob:
    """A normalized unit of work exposed in status snapshots."""

    kind: JobKind
    path: str
    previous_path: str | None = None


@dataclass(frozen=True)
class Failure:
    """A final (retry-exhausted or non-transient) job failure."""

    job: IndexJob
    attempts: int
    error_type: str
    message: str
    timestamp: float


@dataclass(frozen=True)
class IndexingStatus:
    """A cheap, consistent snapshot of coordinator state and progress."""

    running: bool
    stopping: bool
    pending: int
    active: IndexJob | None
    submitted: int
    coalesced: int
    completed: int
    failed: int
    retried: int
    cancelled: int
    discovered: int
    last_failure: Failure | None

    @property
    def idle(self) -> bool:
        return self.pending == 0 and self.active is None


PathCallable = Callable[[str], None]
MoveCallable = Callable[[str, str], None]
ScanCallable = Callable[[str], Iterable[str | Path]]
TransientPredicate = Callable[[BaseException], bool]
StatusCallback = Callable[[IndexingStatus], None]


def _canonical(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve(strict=False))


def _default_is_transient(exc: BaseException) -> bool:
    """Classify common filesystem/database availability failures as retryable."""

    return isinstance(exc, (OSError, sqlite3.OperationalError))


class IndexingService:
    """Serialize, coalesce, and monitor all index mutations.

    Jobs may be queued before :meth:`start`.  ``stop(drain=True)`` finishes all
    queued work, including files discovered by a scan already in the queue.
    ``stop(drain=False)`` drops queued jobs and lets only the currently running
    callable finish (Python threads cannot safely interrupt arbitrary code).

    ``indexer``, ``deleter``, ``mover``, and ``scanner`` receive canonical path
    strings. Supplying these callables avoids importing Ferret's model or
    SQLite extension, which is useful for tests and alternate integrations.
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        model_path: str | Path = "~/ferret/models/bge-small-en",
        *,
        indexer: PathCallable | None = None,
        deleter: PathCallable | None = None,
        mover: MoveCallable | None = None,
        scanner: ScanCallable | None = None,
        supported_extensions: Iterable[str] = SUPPORTED_EXTENSIONS,
        exclude_patterns: Iterable[str] = (),
        max_retries: int = 2,
        retry_delay: float = 0.1,
        retry_backoff: float = 2.0,
        is_transient: TransientPredicate | None = None,
        status_callback: StatusCallback | None = None,
        failure_history: int = 20,
        thread_name: str = "ferret-indexing",
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if retry_delay < 0:
            raise ValueError("retry_delay must be non-negative")
        if retry_backoff < 1:
            raise ValueError("retry_backoff must be at least 1")
        if failure_history < 1:
            raise ValueError("failure_history must be positive")

        self.db_path = _canonical(db_path) if db_path is not None else None
        self.model_path = str(Path(model_path).expanduser())
        self._indexer = indexer or self._make_default_indexer()
        self._deleter = deleter or self._make_default_deleter()
        self._mover = mover or self._make_default_mover()
        self._scanner = scanner or self._default_scanner
        self._extensions = frozenset(
            extension.lower() if extension.startswith(".") else f".{extension.lower()}"
            for extension in supported_extensions
        )
        self._exclude_patterns = frozenset(exclude_patterns)
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._retry_backoff = retry_backoff
        self._is_transient = is_transient or _default_is_transient
        self._status_callback = status_callback
        self._thread_name = thread_name

        self._condition = threading.Condition(threading.RLock())
        # A file key deliberately omits its operation: the newest index/delete
        # request replaces the older intent while preserving queue position.
        self._pending: OrderedDict[tuple[str, str], IndexJob] = OrderedDict()
        self._active: IndexJob | None = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._stopping = False
        self._drain_on_stop = True

        self._submitted = 0
        self._coalesced = 0
        self._completed = 0
        self._failed = 0
        self._retried = 0
        self._cancelled = 0
        self._discovered = 0
        self._failures: deque[Failure] = deque(maxlen=failure_history)

    def _make_default_indexer(self) -> PathCallable:
        if self.db_path is None:
            def missing_db(_path: str) -> None:
                raise ValueError("db_path is required when indexer is not supplied")

            return missing_db

        def index(path: str) -> None:
            from core.indexer import index_file

            index_file(path, self.db_path, self.model_path, raise_on_error=True)

        return index

    def _make_default_deleter(self) -> PathCallable:
        if self.db_path is None:
            def missing_db(_path: str) -> None:
                raise ValueError("db_path is required when deleter is not supplied")

            return missing_db

        def mark_orphaned(path: str) -> None:
            # Match the existing searcher's relocation behavior: retain hashes
            # and chunks so a moved document can be recognized later.
            from core.indexer import _connect

            db = _connect(self.db_path)
            try:
                db.execute(
                    "UPDATE files SET status='orphaned' WHERE path = ?", (path,)
                )
                db.commit()
            except Exception:
                db.rollback()
                raise
            finally:
                db.close()

        return mark_orphaned

    def _make_default_mover(self) -> MoveCallable:
        if self.db_path is None:
            def missing_db(_source: str, _destination: str) -> None:
                raise ValueError("db_path is required when mover is not supplied")

            return missing_db

        def move(source: str, destination: str) -> None:
            from core.indexer import index_file, move_indexed_file

            if not move_indexed_file(source, destination, self.db_path):
                # A file can be renamed before its earlier create event reaches
                # the worker. In that case there is no record to relocate, so
                # index the destination normally instead of dropping it.
                index_file(
                    destination,
                    self.db_path,
                    self.model_path,
                    raise_on_error=True,
                )

        return move

    @staticmethod
    def _default_scanner(folder: str) -> Iterable[Path]:
        root = Path(folder)
        if not root.is_dir():
            raise NotADirectoryError(folder)
        return root.rglob("*")

    def start(self) -> bool:
        """Start the worker, returning ``True`` only when a thread was created."""

        with self._condition:
            if self._thread is not None and self._thread.is_alive():
                return False
            self._stopping = False
            self._drain_on_stop = True
            self._running = True
            self._thread = threading.Thread(
                target=self._run, name=self._thread_name, daemon=True
            )
            self._thread.start()
        self._emit_status()
        return True

    def stop(self, drain: bool = True, timeout: float | None = None) -> bool:
        """Request shutdown and return whether the worker stopped in time."""

        with self._condition:
            thread = self._thread
            if thread is None or not thread.is_alive():
                if not drain and self._pending:
                    self._cancelled += len(self._pending)
                    self._pending.clear()
                self._running = False
                self._stopping = False
                self._condition.notify_all()
                return True
            if thread is threading.current_thread():
                raise RuntimeError("the indexing worker cannot stop itself")
            self._stopping = True
            # Cancellation takes precedence if concurrent/repeated callers use
            # different modes while shutdown is in progress.
            self._drain_on_stop = self._drain_on_stop and drain
            if not self._drain_on_stop and self._pending:
                self._cancelled += len(self._pending)
                self._pending.clear()
            self._condition.notify_all()
        self._emit_status()
        thread.join(timeout)
        return not thread.is_alive()

    def cancel(self, timeout: float | None = None) -> bool:
        """Convenience equivalent of ``stop(drain=False)``."""

        return self.stop(drain=False, timeout=timeout)

    def wait_for_idle(self, timeout: float | None = None) -> bool:
        """Wait until no queued or active job remains, without stopping."""

        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while self._pending or self._active is not None:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True

    def submit_index(self, path: str | Path) -> bool:
        """Queue an index/update.  Returns ``False`` when it coalesced."""

        return self._submit(IndexJob(JobKind.INDEX, _canonical(path)))

    def submit_delete(self, path: str | Path) -> bool:
        """Queue removal from the live index (orphan marking by default)."""

        return self._submit(IndexJob(JobKind.DELETE, _canonical(path)))

    def submit_scan(self, folder: str | Path) -> bool:
        """Queue a recursive folder scan.  Duplicate pending scans coalesce."""

        return self._submit(IndexJob(JobKind.SCAN, _canonical(folder)))

    def enqueue_move(
        self, source: str | Path, destination: str | Path
    ) -> bool:
        """Queue an atomic metadata move without recomputing its embeddings."""

        return self._submit(IndexJob(
            JobKind.MOVE,
            _canonical(destination),
            previous_path=_canonical(source),
        ))

    # Readable aliases for callers that prefer queue/enqueue terminology.
    queue_index = submit_index
    queue_delete = submit_delete
    queue_scan = submit_scan
    enqueue_index = submit_index
    enqueue_file = submit_index
    enqueue_delete = submit_delete
    enqueue_folder_scan = submit_scan
    wait_until_idle = wait_for_idle

    def _submit(self, job: IndexJob, *, internal: bool = False) -> bool:
        key = ("scan" if job.kind is JobKind.SCAN else "file", job.path)
        with self._condition:
            if self._stopping and not internal:
                raise RuntimeError("indexing service is stopping")
            if key in self._pending:
                self._pending[key] = job
                self._coalesced += 1
                accepted = False
            else:
                self._pending[key] = job
                self._submitted += 1
                accepted = True
            self._condition.notify_all()
        self._emit_status()
        return accepted

    @property
    def status(self) -> IndexingStatus:
        with self._condition:
            return self._status_locked()

    @property
    def failures(self) -> tuple[Failure, ...]:
        with self._condition:
            return tuple(self._failures)

    def _status_locked(self) -> IndexingStatus:
        return IndexingStatus(
            running=self._running,
            stopping=self._stopping,
            pending=len(self._pending),
            active=self._active,
            submitted=self._submitted,
            coalesced=self._coalesced,
            completed=self._completed,
            failed=self._failed,
            retried=self._retried,
            cancelled=self._cancelled,
            discovered=self._discovered,
            last_failure=self._failures[-1] if self._failures else None,
        )

    def _emit_status(self) -> None:
        callback = self._status_callback
        if callback is None:
            return
        snapshot = self.status
        try:
            callback(snapshot)
        except Exception:
            # Observability must never take down the indexing worker.
            pass

    def _run(self) -> None:
        try:
            while True:
                with self._condition:
                    while not self._pending and not self._stopping:
                        self._condition.wait()
                    if self._stopping and (
                        not self._drain_on_stop or not self._pending
                    ):
                        return
                    _key, job = self._pending.popitem(last=False)
                    self._active = job
                self._emit_status()

                succeeded = self._execute_with_retry(job)
                with self._condition:
                    if succeeded:
                        self._completed += 1
                    self._active = None
                    self._condition.notify_all()
                self._emit_status()
        finally:
            with self._condition:
                self._running = False
                self._stopping = False
                self._active = None
                self._condition.notify_all()
            self._emit_status()

    def _execute_with_retry(self, job: IndexJob) -> bool:
        attempts = 0
        delay = self._retry_delay
        while True:
            attempts += 1
            try:
                self._execute(job)
                return True
            except Exception as exc:
                with self._condition:
                    cancelling = self._stopping and not self._drain_on_stop
                retry = (
                    not cancelling
                    and attempts <= self._max_retries
                    and self._is_transient(exc)
                )
                if retry:
                    with self._condition:
                        self._retried += 1
                    self._emit_status()
                    if delay:
                        time.sleep(delay)
                    delay *= self._retry_backoff
                    continue

                failure = Failure(
                    job=job,
                    attempts=attempts,
                    error_type=type(exc).__name__,
                    message=str(exc),
                    timestamp=time.time(),
                )
                with self._condition:
                    self._failed += 1
                    self._failures.append(failure)
                self._emit_status()
                return False

    def _execute(self, job: IndexJob) -> None:
        if job.kind is JobKind.INDEX:
            self._indexer(job.path)
        elif job.kind is JobKind.DELETE:
            self._deleter(job.path)
        elif job.kind is JobKind.MOVE:
            if job.previous_path is None:
                raise ValueError("move job is missing its source path")
            self._mover(job.previous_path, job.path)
        else:
            self._scan(job.path)

    def _scan(self, folder: str) -> None:
        candidates = self._scanner(folder)
        for candidate in candidates:
            path = Path(candidate).expanduser()
            if path.suffix.lower() not in self._extensions:
                continue
            if self._exclude_patterns.intersection(path.parts):
                continue
            with self._condition:
                if self._stopping and not self._drain_on_stop:
                    return
                self._discovered += 1
            self._submit(
                IndexJob(JobKind.INDEX, _canonical(path)), internal=True
            )

    def __enter__(self) -> "IndexingService":
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.stop(drain=True)


__all__ = [
    "Failure",
    "IndexJob",
    "IndexingService",
    "IndexingStatus",
    "JobKind",
    "SUPPORTED_EXTENSIONS",
]
