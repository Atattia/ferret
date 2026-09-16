"""Real SQLite/vector regression tests with deterministic embeddings."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sqlite3
import sqlite_vec
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import cli
from core import searcher
from core.indexer import init_db, _serialize_vector
from core.query import parse_query


class ScopedSearchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.db_path = str(self.root / "index.db")
        init_db(self.db_path)
        self.db = sqlite3.connect(self.db_path)
        self.db.enable_load_extension(True)
        sqlite_vec.load(self.db)
        self.db.enable_load_extension(False)
        self.addCleanup(self.db.close)
        self.vector = np.r_[1., np.zeros(383)].astype(np.float32)

    def add_file(self, name, text="Annual revenue report", vector=None):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        file_id = self.db.execute(
            "INSERT INTO files(path, hash, filename) VALUES (?, ?, ?)",
            (str(path), name, path.name),
        ).lastrowid
        chunk_id = self.db.execute(
            "INSERT INTO chunks(file_id, chunk_index, text) VALUES (?, 0, ?)",
            (file_id, text),
        ).lastrowid
        self.db.execute("INSERT INTO chunks_fts(rowid,text) VALUES (?,?)", (chunk_id, text))
        self.db.execute("INSERT INTO vec_chunks(chunk_id,embedding) VALUES (?,?)",
                        (chunk_id, _serialize_vector(self.vector if vector is None else vector)))
        self.db.commit()
        return str(path)

    def test_filter_parser_handles_spaces_and_incomplete_typing(self):
        text, filters = parse_query('growth type:PDF,md in:"~/My Documents"')
        self.assertEqual(text, "growth")
        self.assertEqual(filters.extensions, ("pdf", "md"))
        self.assertEqual(filters.folder, "~/My Documents")
        self.assertEqual(parse_query('growth in:"unfinished')[0], 'growth in:"unfinished')

    def test_filters_are_shared_by_all_routes(self):
        wanted = self.add_file("reports/annual.pdf")
        self.add_file("reports/annual.txt")
        self.add_file("reports-old/annual.pdf")
        with patch.object(searcher, "embed", return_value=self.vector[None, :]):
            results = searcher.search(f'annual type:pdf in:"{self.root / "reports"}"', self.db_path)
        self.assertEqual([r["path"] for r in results], [wanted])
        self.assertEqual(results[0]["matched_by"], ["filename", "fts", "semantic"])

    def test_scoped_semantics_are_not_starved_by_global_neighbors(self):
        for number in range(130):
            self.add_file(f"distractor{number}.txt")
        nearby = np.r_[0.99, 0.1, np.zeros(382)].astype(np.float32)
        nearby /= np.linalg.norm(nearby)
        wanted = self.add_file("target.pdf", vector=nearby)
        with patch.object(searcher, "embed", return_value=self.vector[None, :]):
            results = searcher.search("growth type:pdf", self.db_path, top_k=1, mode="semantic")
        self.assertEqual([r["path"] for r in results], [wanted])

    def test_literal_filename_wildcards_and_folder_boundaries(self):
        wanted = self.add_file("a_b/report_100%.pdf")
        self.add_file("axb/reportX100Y.pdf")
        results = searcher.search(f'report_100% in:"{self.root / "a_b"}"', self.db_path, mode="keyword")
        self.assertEqual([r["path"] for r in results], [wanted])

    def test_filter_only_browses_without_loading_model(self):
        wanted = self.add_file("notes.md")
        self.add_file("notes.txt")
        with patch.object(searcher, "embed", side_effect=AssertionError("model loaded")):
            results = searcher.search("type:md", self.db_path)
        self.assertEqual([r["path"] for r in results], [wanted])

    def test_query_preserves_negation_and_missing_file_does_not_scan(self):
        missing = self.add_file("missing.txt")
        Path(missing).unlink()
        with (patch.object(searcher, "embed", return_value=self.vector[None, :]) as embed,
              patch.object(searcher, "_find_by_hash", side_effect=AssertionError("scanned home"))):
            self.assertEqual(searcher.search("food without nuts", self.db_path, mode="semantic"), [])
        self.assertEqual(embed.call_args.args[0], ["food without nuts"])
        self.assertEqual(self.db.execute("SELECT status FROM files").fetchone()[0], "indexed")

    def test_cli_json_and_stats(self):
        wanted = self.add_file("annual.pdf")
        output = io.StringIO()
        with redirect_stdout(output):
            code = cli.main(["annual", "--db", self.db_path, "--mode", "keyword", "--json"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output.getvalue())["results"][0]["path"], wanted)
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(cli.main(["--db", self.db_path, "--stats"]), 0)
        self.assertEqual(json.loads(output.getvalue())["files_by_status"], {"indexed": 1})

    def test_invalid_mode_and_nonpositive_limit(self):
        with self.assertRaises(ValueError):
            searcher.search("revenue", self.db_path, mode="typo")
        self.assertEqual(searcher.search("revenue", self.db_path, top_k=0), [])


if __name__ == "__main__":
    unittest.main()
