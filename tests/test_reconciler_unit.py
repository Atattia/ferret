"""Dependency-free unit tests for startup filesystem reconciliation."""

import hashlib
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.reconciler import (
    ReconciliationAction,
    StartupReconciler,
    queue_reconciliation_actions,
    reconcile_filesystem,
)
from unittest.mock import MagicMock


def _digest(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _make_db(path: Path, rows=()) -> None:
    db = sqlite3.connect(path)
    try:
        db.execute(
            """
            CREATE TABLE files (
                id INTEGER PRIMARY KEY,
                path TEXT UNIQUE NOT NULL,
                hash TEXT NOT NULL,
                filename TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'indexed'
            )
            """
        )
        db.executemany(
            "INSERT INTO files(id, path, hash, filename, status) VALUES (?, ?, ?, ?, ?)",
            rows,
        )
        db.commit()
    finally:
        db.close()


class ReconcilerTests(unittest.TestCase):
    def test_actions_dispatch_to_single_indexing_service(self):
        service = MagicMock()
        actions = [
            ReconciliationAction("moved", "/new.txt", previous_path="/old.txt"),
            ReconciliationAction("missing", "/gone.txt"),
            ReconciliationAction("excluded", "/cache.txt"),
            ReconciliationAction("changed", "/changed.txt"),
            ReconciliationAction("new", "/new-file.txt"),
        ]
        self.assertEqual(queue_reconciliation_actions(actions, service), 5)
        service.enqueue_move.assert_called_once_with("/old.txt", "/new.txt")
        self.assertEqual(service.enqueue_delete.call_count, 2)
        self.assertEqual(service.enqueue_index.call_count, 2)

    def test_reports_new_changed_missing_and_move_deterministically(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "docs"
            root.mkdir()
            unchanged = root / "unchanged.txt"
            changed = root / "changed.md"
            moved = root / "renamed.docx"
            new = root / "new.pdf"
            unchanged.write_text("same", encoding="utf-8")
            changed.write_text("after", encoding="utf-8")
            moved.write_text("moved content", encoding="utf-8")
            new.write_text("brand new", encoding="utf-8")

            old_moved = root / "old-name.docx"
            missing = root / "deleted.txt"
            db_path = base / "ferret.db"
            _make_db(
                db_path,
                [
                    (1, str(unchanged), _digest(unchanged), unchanged.name, "indexed"),
                    (2, str(changed), "old-hash", changed.name, "indexed"),
                    (3, str(old_moved), _digest(moved), old_moved.name, "indexed"),
                    (4, str(missing), "missing-hash", missing.name, "indexed"),
                ],
            )

            first = reconcile_filesystem([root], db_path)
            second = reconcile_filesystem([root], db_path)

            self.assertEqual(first, second)
            self.assertEqual(
                [(action.kind, Path(action.path).name) for action in first],
                [
                    ("moved", "renamed.docx"),
                    ("missing", "deleted.txt"),
                    ("changed", "changed.md"),
                    ("new", "new.pdf"),
                ],
            )
            self.assertEqual(first[0].previous_path, str(old_moved.resolve()))
            self.assertEqual(first[0].file_id, 3)
            self.assertEqual(first[2].current_hash, _digest(changed))

    def test_exclusions_cover_directory_names_and_glob_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "docs"
            (root / "node_modules").mkdir(parents=True)
            (root / "archive" / "2025").mkdir(parents=True)
            (root / "kept").mkdir()
            ignored_name = root / "node_modules" / "package.txt"
            ignored_glob = root / "archive" / "2025" / "notes.md"
            ignored_file = root / "kept" / "draft-secret.txt"
            included = root / "kept" / "notes.txt"
            for path in (ignored_name, ignored_glob, ignored_file, included):
                path.write_text(path.name, encoding="utf-8")

            db_path = base / "ferret.db"
            # An old record below an excluded directory is outside scope and
            # must not appear to have been deleted.
            excluded_missing = root / "node_modules" / "gone.txt"
            _make_db(
                db_path,
                [(1, str(excluded_missing), "old", excluded_missing.name, "indexed")],
            )

            actions = reconcile_filesystem(
                [root],
                db_path,
                exclude_patterns=["node_modules", "archive/**", "*-secret.txt"],
            )

            self.assertEqual(
                [(action.kind, action.path) for action in actions],
                [
                    ("excluded", str(excluded_missing.resolve())),
                    ("new", str(included.resolve())),
                ],
            )

    def test_shallow_user_documents_are_queued_before_deep_dependencies(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "docs"
            deep = root / "cache" / "package" / "docs"
            deep.mkdir(parents=True)
            shallow = root / "PLAN.md"
            dependency = deep / "README.md"
            shallow.write_text("plan", encoding="utf-8")
            dependency.write_text("package", encoding="utf-8")
            db_path = base / "ferret.db"
            _make_db(db_path)

            actions = reconcile_filesystem([root], db_path)

            self.assertEqual(
                [Path(action.path).name for action in actions],
                ["PLAN.md", "README.md"],
            )

    def test_missing_configured_folder_does_not_orphan_its_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            available = base / "available"
            available.mkdir()
            unavailable = base / "offline-drive"
            db_path = base / "ferret.db"
            _make_db(
                db_path,
                [
                    (
                        1,
                        str(unavailable / "document.txt"),
                        "stored",
                        "document.txt",
                        "indexed",
                    )
                ],
            )

            actions = reconcile_filesystem([unavailable], db_path)
            mixed_actions = reconcile_filesystem([available, unavailable], db_path)

            self.assertEqual(actions, [])
            self.assertEqual(mixed_actions, [])

    def test_hashing_is_injectable_and_new_files_are_not_hashed_without_moves(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "docs"
            root.mkdir()
            known = root / "known.txt"
            new = root / "new.txt"
            known.write_text("known", encoding="utf-8")
            new.write_text("new", encoding="utf-8")
            db_path = base / "ferret.db"
            _make_db(
                db_path,
                [(1, str(known), "known-digest", known.name, "indexed")],
            )
            calls = []

            def fake_hasher(path):
                calls.append(Path(path).name)
                return "known-digest"

            # A single Path is accepted in addition to the configuration's
            # normal list-of-paths form.
            actions = StartupReconciler(db_path, hasher=fake_hasher).reconcile(root)

            self.assertEqual(calls, ["known.txt"])
            self.assertEqual([(action.kind, Path(action.path).name) for action in actions], [("new", "new.txt")])
            self.assertIsNone(actions[0].current_hash)

    def test_non_indexed_record_is_changed_even_when_content_hash_matches(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "docs"
            root.mkdir()
            restored = root / "restored.txt"
            restored.write_text("back", encoding="utf-8")
            stored_hash = _digest(restored)
            db_path = base / "ferret.db"
            _make_db(
                db_path,
                [(1, str(restored), stored_hash, restored.name, "orphaned")],
            )

            actions = reconcile_filesystem([root], db_path)

            self.assertEqual(len(actions), 1)
            self.assertEqual(actions[0].kind, "changed")
            self.assertEqual(actions[0].stored_hash, stored_hash)
            self.assertEqual(actions[0].current_hash, stored_hash)

    def test_unsupported_files_and_case_are_handled(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "docs"
            root.mkdir()
            supported = root / "README.MD"
            unsupported = root / "image.png"
            supported.write_text("text", encoding="utf-8")
            unsupported.write_bytes(b"png")
            db_path = base / "ferret.db"
            _make_db(db_path)

            actions = reconcile_filesystem([root], db_path)

            self.assertEqual(len(actions), 1)
            self.assertEqual(actions[0].path, str(supported.resolve()))


if __name__ == "__main__":
    unittest.main()
