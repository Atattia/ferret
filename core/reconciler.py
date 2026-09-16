"""Read-only startup reconciliation between configured folders and the index.

The reconciler deliberately does not import the indexer (and therefore does not
load sqlite-vec, ONNX, or Qt).  Its result is a small, deterministic list of
actions which a caller can hand to whichever indexing coordinator it uses.
"""

from __future__ import annotations

from dataclasses import dataclass
import fnmatch
import os
from pathlib import Path, PurePosixPath
import sqlite3
from typing import Callable, Iterable, Sequence

from core.hasher import hash_file


SUPPORTED_EXTENSIONS = frozenset({".pdf", ".docx", ".txt", ".md"})
ACTION_NEW = "new"
ACTION_CHANGED = "changed"
ACTION_MOVED = "moved"
ACTION_MISSING = "missing"
ACTION_EXCLUDED = "excluded"


@dataclass(frozen=True)
class ReconciliationAction:
    """One difference between the filesystem and the ``files`` table.

    ``path`` is the on-disk path for ``new``/``changed``/``moved`` actions and
    the last-known path for ``missing`` actions.  A move additionally exposes
    its old path as ``previous_path``.  The reconciler itself never mutates the
    database.
    """

    kind: str
    path: str
    file_id: int | None = None
    previous_path: str | None = None
    stored_hash: str | None = None
    current_hash: str | None = None


@dataclass(frozen=True)
class _FileRecord:
    file_id: int
    path: str
    stored_hash: str
    status: str


Hasher = Callable[[str | Path], str]


def _normalise_patterns(patterns: Iterable[str] | None) -> tuple[str, ...]:
    if isinstance(patterns, str):
        patterns = (patterns,)
    result = []
    for pattern in patterns or ():
        if not isinstance(pattern, str):
            continue
        pattern = pattern.strip().replace("\\", "/")
        while pattern.startswith("./"):
            pattern = pattern[2:]
        pattern = pattern.strip("/")
        if pattern:
            result.append(pattern)
    return tuple(dict.fromkeys(result))


def _matches_exclusion(relative_path: Path, patterns: Sequence[str]) -> bool:
    """Match both directory names and path-shaped glob patterns."""
    relative = relative_path.as_posix().strip("/")
    if not relative or relative == ".":
        return False
    parts = PurePosixPath(relative).parts

    for pattern in patterns:
        # A name-only pattern applies to any component.  This covers the
        # common configuration values (``.git`` and ``node_modules``) as well
        # as globbed names such as ``*.cache``.
        if "/" not in pattern:
            if any(fnmatch.fnmatchcase(part, pattern) for part in parts):
                return True
            continue

        # Path-shaped patterns are relative to a configured root.  ``match``
        # supplies pathlib's ** semantics; fnmatch also handles conventional
        # forms such as ``private/*`` in the unsurprising way users expect.
        if fnmatch.fnmatchcase(relative, pattern):
            return True
        try:
            if PurePosixPath(relative).match(pattern):
                return True
        except ValueError:
            # An invalid glob is simply a non-match, like an unknown setting.
            continue
    return False


def _normalise_roots(folders: Iterable[str | Path] | None) -> tuple[Path, ...]:
    if isinstance(folders, (str, Path)):
        folders = (folders,)
    roots: set[Path] = set()
    for folder in folders or ():
        try:
            path = Path(folder).expanduser().resolve()
            if path.is_dir():
                roots.add(path)
        except (TypeError, ValueError, OSError):
            continue
    return tuple(sorted(roots, key=lambda path: str(path)))


def _relative_to_any(path: Path, roots: Sequence[Path]) -> tuple[Path, ...]:
    relatives = []
    for root in roots:
        try:
            relatives.append(path.relative_to(root))
        except ValueError:
            pass
    return tuple(relatives)


def _is_in_scope(path: Path, roots: Sequence[Path], patterns: Sequence[str]) -> bool:
    relatives = _relative_to_any(path, roots)
    return bool(relatives) and not any(
        _matches_exclusion(relative, patterns) for relative in relatives
    )


def _scan(
    roots: Sequence[Path], patterns: Sequence[str], extensions: frozenset[str]
) -> tuple[dict[str, Path], tuple[Path, ...]]:
    files: dict[str, Path] = {}
    inaccessible: set[Path] = set()

    def note_error(error: OSError) -> None:
        if error.filename:
            try:
                inaccessible.add(Path(error.filename).resolve())
            except (OSError, ValueError):
                pass

    for root in roots:
        for directory, directory_names, file_names in os.walk(
            root, topdown=True, onerror=note_error, followlinks=False
        ):
            directory_path = Path(directory)
            try:
                relative_directory = directory_path.relative_to(root)
            except ValueError:
                continue

            # Pruning excluded directories avoids both needless traversal and
            # accidental actions for their descendants.
            directory_names[:] = sorted(
                name
                for name in directory_names
                if not _matches_exclusion(relative_directory / name, patterns)
            )
            for name in sorted(file_names):
                lexical_path = directory_path / name
                if lexical_path.suffix.lower() not in extensions:
                    continue
                relative_path = relative_directory / name
                if _matches_exclusion(relative_path, patterns):
                    continue
                try:
                    path = lexical_path.resolve()
                    if path.is_file() and _is_in_scope(path, roots, patterns):
                        files[str(path)] = path
                except (OSError, ValueError):
                    continue

    return files, tuple(sorted(inaccessible, key=lambda path: str(path)))


def _under_any(path: Path, prefixes: Sequence[Path]) -> bool:
    for prefix in prefixes:
        try:
            path.relative_to(prefix)
            return True
        except ValueError:
            pass
    return False


def _load_records(db_path: str | Path) -> list[_FileRecord]:
    # A plain sqlite connection is sufficient: reading a normal table does not
    # require loading the sqlite-vec extension used by the search/index paths.
    db = sqlite3.connect(str(Path(db_path).expanduser()))
    try:
        rows = db.execute(
            "SELECT id, path, hash, status FROM files ORDER BY path, id"
        ).fetchall()
    finally:
        db.close()
    return [
        _FileRecord(int(file_id), str(path), str(stored_hash), str(status))
        for file_id, path, stored_hash, status in rows
    ]


def reconcile_filesystem(
    folders: str | Path | Iterable[str | Path] | None,
    db_path: str | Path,
    *,
    exclude_patterns: Iterable[str] | None = None,
    hasher: Hasher = hash_file,
    supported_extensions: Iterable[str] = SUPPORTED_EXTENSIONS,
) -> list[ReconciliationAction]:
    """Return deterministic startup actions without changing the database.

    Actions use the kinds ``new``, ``changed``, ``moved``, and ``missing``.
    Missing configured folders are ignored: records below an unavailable root
    are not reported missing merely because a drive is temporarily offline.

    The current schema has no mtime or size column, so known files must be
    hashed to establish whether their content changed.  New files are only
    hashed when there are missing records with which they might form moves.
    Every discovered path is hashed at most once.
    """
    roots = _normalise_roots(folders)
    if not roots:
        return []

    patterns = _normalise_patterns(exclude_patterns)
    if isinstance(supported_extensions, str):
        supported_extensions = (supported_extensions,)
    extensions = frozenset(
        extension.lower() if extension.startswith(".") else f".{extension.lower()}"
        for extension in supported_extensions
        if isinstance(extension, str) and extension
    )
    disk_files, inaccessible = _scan(roots, patterns, extensions)

    records = []
    excluded_records = []
    for record in _load_records(db_path):
        try:
            path = Path(record.path).expanduser().resolve()
        except (OSError, ValueError):
            continue
        if path.suffix.lower() not in extensions:
            continue
        relatives = _relative_to_any(path, roots)
        if not relatives:
            continue
        if any(_matches_exclusion(relative, patterns) for relative in relatives):
            excluded_records.append((record, path))
        else:
            records.append((record, path))

    records_by_path = {str(path): record for record, path in records}
    disk_paths = set(disk_files)
    known_paths = set(records_by_path)
    present_paths = sorted(disk_paths & known_paths)
    new_paths = sorted(disk_paths - known_paths)
    missing_paths = sorted(
        path
        for path in known_paths - disk_paths
        if not _under_any(Path(path), inaccessible)
    )

    hashes: dict[str, str] = {}

    def get_hash(path: str) -> str:
        if path not in hashes:
            try:
                hashes[path] = hasher(disk_files[path]) or ""
            except (OSError, ValueError):
                hashes[path] = ""
        return hashes[path]

    actions: list[ReconciliationAction] = []

    for record, path in excluded_records:
        if record.status == "indexed":
            actions.append(ReconciliationAction(
                kind=ACTION_EXCLUDED,
                path=str(path),
                file_id=record.file_id,
                stored_hash=record.stored_hash,
            ))

    for path in present_paths:
        record = records_by_path[path]
        current_hash = get_hash(path)
        if current_hash and (
            current_hash != record.stored_hash or record.status != "indexed"
        ):
            actions.append(
                ReconciliationAction(
                    kind=ACTION_CHANGED,
                    path=path,
                    file_id=record.file_id,
                    stored_hash=record.stored_hash,
                    current_hash=current_hash,
                )
            )

    # A matching content hash turns a missing+new pair into a move.  Sorting
    # both sides makes even duplicate-content matches stable across launches.
    missing_by_hash: dict[str, list[str]] = {}
    for path in missing_paths:
        stored_hash = records_by_path[path].stored_hash
        if stored_hash:
            missing_by_hash.setdefault(stored_hash, []).append(path)

    new_by_hash: dict[str, list[str]] = {}
    if missing_by_hash:
        for path in new_paths:
            current_hash = get_hash(path)
            if current_hash in missing_by_hash:
                new_by_hash.setdefault(current_hash, []).append(path)

    moved_old: set[str] = set()
    moved_new: set[str] = set()
    for content_hash in sorted(set(missing_by_hash) & set(new_by_hash)):
        old_group = sorted(missing_by_hash[content_hash])
        new_group = sorted(new_by_hash[content_hash])
        for old_path, new_path in zip(old_group, new_group):
            record = records_by_path[old_path]
            moved_old.add(old_path)
            moved_new.add(new_path)
            actions.append(
                ReconciliationAction(
                    kind=ACTION_MOVED,
                    path=new_path,
                    file_id=record.file_id,
                    previous_path=old_path,
                    stored_hash=record.stored_hash,
                    current_hash=content_hash,
                )
            )

    for path in missing_paths:
        if path in moved_old:
            continue
        record = records_by_path[path]
        actions.append(
            ReconciliationAction(
                kind=ACTION_MISSING,
                path=path,
                file_id=record.file_id,
                stored_hash=record.stored_hash,
            )
        )

    for path in new_paths:
        if path in moved_new:
            continue
        current_hash = hashes.get(path) or None
        actions.append(
            ReconciliationAction(
                kind=ACTION_NEW, path=path, current_hash=current_hash
            )
        )

    action_order = {
        ACTION_MOVED: 0,
        ACTION_EXCLUDED: 1,
        ACTION_MISSING: 2,
        ACTION_CHANGED: 3,
        ACTION_NEW: 4,
    }

    def source_depth(path: str) -> int:
        candidate = Path(path)
        depths = []
        for root in roots:
            try:
                depths.append(len(candidate.relative_to(root).parts))
            except ValueError:
                pass
        return min(depths) if depths else 10_000

    return sorted(
        actions,
        key=lambda action: (
            action_order[action.kind],
            source_depth(action.path),
            action.previous_path or "",
            action.path,
            action.file_id if action.file_id is not None else -1,
        ),
    )


class StartupReconciler:
    """Convenient configured wrapper around :func:`reconcile_filesystem`."""

    def __init__(
        self,
        db_path: str | Path,
        *,
        exclude_patterns: Iterable[str] | None = None,
        hasher: Hasher = hash_file,
        supported_extensions: Iterable[str] = SUPPORTED_EXTENSIONS,
    ) -> None:
        self.db_path = db_path
        if isinstance(exclude_patterns, str):
            exclude_patterns = (exclude_patterns,)
        if isinstance(supported_extensions, str):
            supported_extensions = (supported_extensions,)
        self.exclude_patterns = tuple(exclude_patterns or ())
        self.hasher = hasher
        self.supported_extensions = tuple(supported_extensions)

    def reconcile(
        self, folders: str | Path | Iterable[str | Path] | None
    ) -> list[ReconciliationAction]:
        return reconcile_filesystem(
            folders,
            self.db_path,
            exclude_patterns=self.exclude_patterns,
            hasher=self.hasher,
            supported_extensions=self.supported_extensions,
        )


def queue_reconciliation_actions(actions, indexing_service) -> int:
    """Submit reconciliation output to an indexing coordinator."""
    count = 0
    for action in actions:
        if action.kind == ACTION_MOVED:
            if not action.previous_path:
                raise ValueError("moved reconciliation action has no previous path")
            indexing_service.enqueue_move(action.previous_path, action.path)
        elif action.kind in {ACTION_MISSING, ACTION_EXCLUDED}:
            indexing_service.enqueue_delete(action.path)
        elif action.kind in {ACTION_NEW, ACTION_CHANGED}:
            indexing_service.enqueue_index(action.path)
        else:
            raise ValueError(f"unknown reconciliation action: {action.kind}")
        count += 1
    return count
