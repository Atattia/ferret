"""Integration tests for chunk-level embedding reuse and indexed moves."""

from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.indexer import _connect, index_file, init_db, move_indexed_file


class _Encoding:
    def __init__(self, count):
        self.ids = list(range(count))


class _Tokenizer:
    truncation = None

    def encode(self, text):
        return _Encoding(len(text.split()))


class IndexReuseTests(unittest.TestCase):
    def test_modified_file_only_embeds_changed_chunks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "notes.txt"
            path.write_text("placeholder", encoding="utf-8")
            db_path = str(root / "ferret.db")
            init_db(db_path)

            first = " ".join(f"alpha{i}" for i in range(300))
            second = " ".join(f"beta{i}" for i in range(300))
            replacement = " ".join(f"gamma{i}" for i in range(300))
            extracted = [f"{first}\n\n{second}", f"{first}\n\n{replacement}"]
            hashes = ["file-v1", "file-v2"]
            embedded_batches = []

            def fake_embed(texts, _model_path):
                embedded_batches.append(list(texts))
                return np.ones((len(texts), 384), dtype=np.float32)

            with (
                patch("core.indexer.extract", side_effect=extracted),
                patch("core.indexer.hash_file", side_effect=hashes),
                patch("core.indexer._get_session", return_value=(_Tokenizer(), object())),
                patch("core.indexer.embed", side_effect=fake_embed),
            ):
                self.assertTrue(index_file(path, db_path))
                self.assertTrue(index_file(path, db_path))

            self.assertEqual(len(embedded_batches[0]), 2)
            self.assertEqual(embedded_batches[0][0], first)
            self.assertIn(second, embedded_batches[0][1])
            self.assertEqual(len(embedded_batches[1]), 1)
            self.assertIn(replacement, embedded_batches[1][0])
            db = _connect(db_path)
            try:
                rows = db.execute(
                    "SELECT chunk_index, text, content_hash FROM chunks ORDER BY chunk_index"
                ).fetchall()
                vector_count = db.execute("SELECT count(*) FROM vec_chunks").fetchone()[0]
            finally:
                db.close()
            self.assertEqual(rows[0][1], first)
            self.assertIn(replacement, rows[1][1])
            self.assertTrue(all(row[2] for row in rows))
            self.assertEqual(vector_count, 2)

    def test_move_preserves_file_and_chunk_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_path = root / "old.txt"
            new_path = root / "new.txt"
            old_path.write_text("text", encoding="utf-8")
            db_path = str(root / "ferret.db")
            init_db(db_path)

            db = sqlite3.connect(db_path)
            try:
                cursor = db.execute(
                    "INSERT INTO files(path, hash, filename) VALUES (?,?,?)",
                    (str(old_path.resolve()), "digest", old_path.name),
                )
                file_id = cursor.lastrowid
                cursor = db.execute(
                    "INSERT INTO chunks(file_id, chunk_index, text) VALUES (?,?,?)",
                    (file_id, 0, "text"),
                )
                chunk_id = cursor.lastrowid
                db.commit()
            finally:
                db.close()

            old_path.rename(new_path)
            self.assertTrue(move_indexed_file(old_path, new_path, db_path))

            db = sqlite3.connect(db_path)
            try:
                file_row = db.execute(
                    "SELECT id, path, filename, status FROM files"
                ).fetchone()
                stored_chunk_id = db.execute("SELECT id FROM chunks").fetchone()[0]
            finally:
                db.close()
            self.assertEqual(
                file_row,
                (file_id, str(new_path.resolve()), "new.txt", "indexed"),
            )
            self.assertEqual(stored_chunk_id, chunk_id)


if __name__ == "__main__":
    unittest.main()
