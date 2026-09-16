"""Tests for safe embedding-pipeline index migration."""

from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.indexer import init_db


class IndexMetadataTests(unittest.TestCase):
    def test_changed_embedding_pipeline_retries_until_stale_files_are_rebuilt(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "index.db")
            self.assertFalse(init_db(db_path))

            db = sqlite3.connect(db_path)
            db.execute(
                "INSERT INTO files(path, hash, filename) VALUES (?,?,?)",
                ("/tmp/report.txt", "old-hash", "report.txt"),
            )
            file_id = db.execute(
                "SELECT id FROM files WHERE path='/tmp/report.txt'"
            ).fetchone()[0]
            db.execute(
                "INSERT INTO chunks(file_id, chunk_index, text) VALUES (?,?,?)",
                (file_id, 0, "old embedded text"),
            )
            db.execute(
                "UPDATE index_metadata SET value='legacy' "
                "WHERE key='embedding_pipeline_version'"
            )
            db.commit()
            db.close()

            self.assertTrue(init_db(db_path))
            db = sqlite3.connect(db_path)
            row = db.execute(
                "SELECT hash, status FROM files WHERE id=?", (file_id,)
            ).fetchone()
            db.close()

            self.assertEqual(row, ("", "stale"))
            self.assertTrue(init_db(db_path))

            db = sqlite3.connect(db_path)
            db.execute(
                "UPDATE files SET hash='new-hash', status='indexed' WHERE id=?",
                (file_id,),
            )
            db.commit()
            db.close()
            self.assertFalse(init_db(db_path))


if __name__ == "__main__":
    unittest.main()
