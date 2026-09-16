from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import numpy as np

from benchmarks.corpus import materialize
from core import indexer, searcher
from core.language import normalize, normalize_with_offsets, lexical_text, passage_preview, is_rtl
from core.models import ModelSpec, model_spec, check_index_model
from core.maintenance import atomic_json, config_lock, rebuild


class ArabicTests(unittest.TestCase):
    def test_normalization_preserves_original_and_meaningful_letters(self):
        original = "إِجــازة ١٢۳ وكتاب يکفی"
        self.assertEqual(normalize(original), "اجازة 123 وكتاب يكفي")
        self.assertNotEqual(normalize("مدرسة"), normalize("مدرسه"))
        self.assertNotEqual(normalize("مسؤول"), normalize("مسئول"))
        self.assertTrue(is_rtl(original))
        self.assertFalse(is_rtl("report ١٢"))

    def test_offsets_and_preview_refer_to_original_text(self):
        text = "x " * 200 + "الإِجَازَة السنوية"
        normalized, offsets = normalize_with_offsets(text)
        start = normalized.index("اجازة")
        self.assertEqual(text[offsets[start]], "إ")
        self.assertIn("الإِجَازَة", passage_preview(text, "اجازة"))

    def test_articles_have_variants_without_destructive_stemming(self):
        self.assertIn("مدارس", lexical_text("بالمدارس"))
        self.assertTrue(lexical_text("بالمدارس").startswith("بالمدارس"))
        self.assertEqual(lexical_text("والد"), "والد")

    def test_arabic_and_mixed_queries_reach_semantic_search(self):
        for query in ("ضغط العمل", "الناس تعبانة من الشغل", "server تعطل"):
            self.assertTrue(searcher._semantic_query_is_usable(query))

    def test_real_normalized_index_and_deletion(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "index.db")
            path = Path(tmp) / "record.txt"
            path.touch()
            indexer.init_db(db_path)
            db = indexer._connect(db_path)
            file_id = db.execute("INSERT INTO files(path,hash,filename) VALUES (?,?,?)",
                                 (str(path), "x", path.name)).lastrowid
            chunk_id = db.execute("INSERT INTO chunks(file_id,chunk_index,text) VALUES (?,0,?)",
                                  (file_id, "الإِجَازَة بالمدارس ٨٠٤")).lastrowid
            db.commit()
            indexer.rebuild_fts(db_path)
            result = searcher.search("اجازة مدارس 804", db_path, mode="keyword")
            self.assertEqual(result[0]["path"], str(path))
            self.assertIn("الإِجَازَة", result[0]["snippet"])
            db.execute("DELETE FROM chunks WHERE id=?", (chunk_id,))
            db.commit()
            self.assertEqual(db.execute("SELECT count(*) FROM chunks_ar_fts").fetchone()[0], 0)
            db.close()

    def test_arabic_filename_variants_and_copy_deduplication(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = str(root / "index.db")
            indexer.init_db(db_path)
            db = indexer._connect(db_path)
            for name in ("إجازة_٢٠٢٦.txt", "اجازة_2026.txt"):
                path = root / name
                path.touch()
                db.execute("INSERT INTO files(path,hash,filename) VALUES (?,?,?)", (str(path), "same-content", name))
            db.commit()
            db.close()
            result = searcher.search("اجازة 2026", db_path, mode="keyword")
            self.assertEqual(len(result), 1)
            self.assertEqual(len(result[0]["copies"]), 2)


class ModelSafetyTests(unittest.TestCase):
    def test_successful_rebuild_switches_only_after_validation_and_keeps_old_index(self):
        from core.hasher import hash_file
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            source = root / "documents"
            source.mkdir()
            (source / "note.txt").write_text("A bilingual search test")
            original_db = root / "original.db"
            original_db.write_bytes(b"old database retained")
            config = root / "settings.json"
            original = dict(db_path=str(original_db), indexed_folders=[str(source)], ocr_languages="eng")
            atomic_json(config, original)

            def index_document(path, db_path, model_path, **kwargs):
                db = indexer._connect(db_path)
                file_id = db.execute("INSERT INTO files(path,hash,filename) VALUES (?,?,?)",
                                     (path, hash_file(path), Path(path).name)).lastrowid
                chunk_id = db.execute("INSERT INTO chunks(file_id,chunk_index,text) VALUES (?,0,'test')",
                                      (file_id,)).lastrowid
                db.execute("INSERT INTO vec_chunks(chunk_id,embedding) VALUES (?,?)",
                           (chunk_id, indexer._serialize_vector(np.ones(384, np.float32) / np.sqrt(384))))
                db.commit()
                db.close()

            with patch("core.maintenance.load_model"), patch.object(indexer, "index_file", side_effect=index_document):
                report = rebuild(config, str(root / "model"), str(root / "new.db"), activate=True)
            self.assertTrue(report["activated"])
            self.assertEqual(json.loads(config.read_text())["db_path"], str(root / "new.db"))
            self.assertEqual(json.loads(Path(str(config) + ".previous").read_text()), original)
            self.assertEqual(original_db.read_bytes(), b"old database retained")

    def test_model_mismatch_refuses_writes_and_preserves_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first, second = root / "first", root / "second"
            for folder, name in ((first, "one"), (second, "two")):
                atomic_json(folder / "ferret-model.json", dict(ModelSpec(name=name).__dict__))
            db = str(root / "index.db")
            indexer.init_db(db, str(first))
            with self.assertRaisesRegex(ValueError, "another model"):
                indexer.init_db(db, str(second))
            conn = indexer._connect(db)
            check_index_model(conn, str(first))
            self.assertEqual(conn.execute("PRAGMA integrity_check").fetchone()[0], "ok")
            conn.close()

    def test_schema_uses_declared_dimensions(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model"
            atomic_json(model / "ferret-model.json", dict(ModelSpec(dimensions=768).__dict__))
            db = str(Path(tmp) / "index.db")
            indexer.init_db(db, str(model))
            conn = sqlite3.connect(db)
            sql = conn.execute("SELECT sql FROM sqlite_master WHERE name='vec_chunks'").fetchone()[0]
            self.assertIn("768", sql)
            conn.close()

    def test_failed_rebuild_does_not_switch_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "settings.json"
            original = dict(db_path=str(root / "original.db"), indexed_folders=[str(root)])
            atomic_json(config, original)
            with patch("core.maintenance.load_model", side_effect=RuntimeError("bad model")):
                with self.assertRaises(RuntimeError):
                    rebuild(config, str(root / "model"), str(root / "new.db"), activate=True)
            self.assertEqual(json.loads(config.read_text()), original)

    def test_exclusive_maintenance_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "settings.json"
            with config_lock(config):
                with self.assertRaises(RuntimeError):
                    with config_lock(config):
                        pass

    def test_missing_model_is_diagnostic_not_no_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "index.db")
            indexer.init_db(path)
            warnings = []
            searcher.search("مرحبا", path, model_path=str(Path(tmp) / "missing"), diagnostics=warnings)
            self.assertTrue(any("Semantic search unavailable" in message for message in warnings))


class RankingTests(unittest.TestCase):
    def test_calibration_counts_unscored_short_queries_as_retained(self):
        from benchmarks.calibrate import fit
        report = dict(split="development", reranker="local", overall=dict(warnings=0), rows=[
            dict(split="development", relevant=["yes"], results=[dict(filename="yes", rerank_score=5.)]),
            dict(split="development", relevant=[], results=[dict(filename="other", rerank_score=None)]),
        ])
        chosen, _ = fit(report)
        self.assertEqual(chosen["hit_at_5"], 1.)
        self.assertEqual(chosen["false_positive_rate"], 1.)

    def test_calibration_rejects_heldout_data_and_model_mismatch(self):
        from benchmarks.calibrate import fit
        from core.models import calibrated_threshold
        with self.assertRaises(ValueError):
            fit(dict(split="heldout", reranker="example"))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calibration.json"
            atomic_json(path, dict(embedding_fingerprint="wrong", reranker_fingerprint="wrong", threshold=0))
            with self.assertRaises(ValueError):
                calibrated_threshold(path, tmp, tmp)

    def test_reranker_can_promote_candidate_and_select_source_passage(self):
        candidates = [dict(path="a", filename="a", snippet="irrelevant", score=1., matched_by=["semantic"]),
                      dict(path="b", filename="b", snippet="first", score=.5, matched_by=["semantic"],
                           passages=[dict(text="first", page=1), dict(text="supporting evidence", page=7)])]
        model = MagicMock()
        model.rerank.return_value = [-4., -2., 5.]
        with patch.object(searcher, "load_model", return_value=model):
            results = searcher._rerank("question", candidates, "local", minimum=0.)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["path"], "b")
        self.assertEqual(results[0]["page"], 7)
        self.assertEqual(results[0]["snippet"], "supporting evidence")

    def test_benchmark_groups_do_not_leak_between_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            queries = materialize(Path(tmp))
        self.assertEqual(len(queries), 208)
        groups = {}
        for query in queries:
            self.assertEqual(groups.setdefault(query["group"], query["split"]), query["split"])
        self.assertIn("egy→en", {query["language"] for query in queries})


if __name__ == "__main__":
    unittest.main()
