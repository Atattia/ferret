"""Dependency-light unit tests for hybrid result fusion."""

from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import sqlite_vec

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import searcher
from core.indexer import _serialize_vector, init_db
from core.searcher import _fuse_ranked_results


def _hit(path: str, snippet: str = "") -> dict:
    return {
        "filename": Path(path).name,
        "path": path,
        "snippet": snippet,
        "score": 0.5,
    }


class SearchFusionTests(unittest.TestCase):
    def test_partial_keyword_evidence_cannot_cast_a_full_vote(self):
        intended = _hit("/tmp/meaning.txt")
        incidental = dict(_hit("/tmp/incidental.txt", "tracking"), query_coverage=.2)
        semantic_incidental = _hit("/tmp/incidental.txt", "semantic passage")
        results = _fuse_ranked_results([
            ("semantic", [intended, semantic_incidental]),
            ("fts", [incidental]),
        ], 2)
        # An incidental keyword still adds evidence, but not a full vote.
        evidence = next(e for r in results for e in r["evidence"] if e["route"] == "fts")
        self.assertEqual(evidence["query_coverage"], .2)
        weighted = next(r["score"] for r in results if r["path"] == incidental["path"])
        full_results = _fuse_ranked_results([
            ("semantic", [intended, semantic_incidental]),
            ("fts", [dict(incidental, query_coverage=1.)]),
        ], 2)
        intended_weighted = next(r["score"] for r in results if r["path"] == intended["path"])
        intended_full = next(r["score"] for r in full_results if r["path"] == intended["path"])
        self.assertGreater(intended_weighted, intended_full)
        self.assertGreater(weighted, 0)

    def test_keyword_coverage_is_language_agnostic_and_bounded(self):
        groups = searcher._coverage_groups("warehouse order tracking")
        self.assertAlmostEqual(searcher._lexical_coverage("tracking tracking tracking", groups), 1/3)
        self.assertEqual(searcher._lexical_coverage("warehouse order tracking", groups), 1.)
        self.assertEqual(searcher._lexical_coverage("unrelated", groups), 0.)
        self.assertEqual(searcher._lexical_coverage("بالمدارس ٢٠٢٦", searcher._coverage_groups("مدارس 2026")), 1.)
        self.assertEqual(searcher._coverage_groups("tracking tracking"), searcher._coverage_groups("tracking"))
        self.assertEqual(searcher._lexical_coverage("résumé café", searcher._coverage_groups("resume cafe")), 1.)
        self.assertNotEqual(searcher._fold_fts_word("مسؤول"), searcher._fold_fts_word("مسئول"))

    def test_zero_coverage_results_do_not_divide_by_zero(self):
        results = _fuse_ranked_results([("fts", [dict(_hit("/tmp/zero.txt"), query_coverage=0.)])], 1)
        self.assertEqual(results[0]["score"], 0.)

    def test_full_keyword_match_outranks_rare_incidental_match(self):
        results = _fuse_ranked_results([("fts", [
            dict(_hit("/tmp/incidental.txt"), query_coverage=.25),
            dict(_hit("/tmp/complete.txt"), query_coverage=1.),
        ])], 2)
        self.assertEqual(results[0]["path"], "/tmp/complete.txt")

    def test_resume_intent_finds_structural_cv_without_resume_word(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = str(root / "index.db")
            cv = root / "Mahmoud_Attia.pdf"
            contract = root / "contract.pdf"
            cv.touch()
            contract.touch()
            init_db(db_path)

            db = sqlite3.connect(db_path)
            cv_id = db.execute(
                "INSERT INTO files(path, hash, filename) VALUES (?,?,?)",
                (str(cv), "cv", cv.name),
            ).lastrowid
            contract_id = db.execute(
                "INSERT INTO files(path, hash, filename) VALUES (?,?,?)",
                (str(contract), "contract", contract.name),
            ).lastrowid
            db.execute(
                "INSERT INTO chunks(file_id, chunk_index, text) VALUES (?,?,?)",
                (cv_id, 0, "Mahmoud Attia\nExperience\nSoftware Engineer\nEducation\nProjects"),
            )
            db.execute(
                "INSERT INTO chunks(file_id, chunk_index, text) VALUES (?,?,?)",
                (contract_id, 0, "Employment agreement and prior experience"),
            )
            db.commit()
            db.close()

            with patch.object(searcher, "_semantic_search", return_value=[]):
                results = searcher.search("CV", db_path, top_k=5)

        self.assertEqual([result["filename"] for result in results], [cv.name])
        self.assertEqual(results[0]["matched_by"], ["doctype"])

    def test_real_database_hybrid_search_without_model_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = str(root / "index.db")
            report = root / "quarterly.txt"
            distractor = root / "cooking.txt"
            report.write_text("Quarterly revenue increased strongly.", encoding="utf-8")
            distractor.write_text("Simmer the onions in olive oil.", encoding="utf-8")
            init_db(db_path)

            db = sqlite3.connect(db_path)
            db.enable_load_extension(True)
            sqlite_vec.load(db)
            db.enable_load_extension(False)
            for path, text, vector in (
                (report, report.read_text(), np.r_[1.0, np.zeros(383)]),
                (distractor, distractor.read_text(), np.r_[0.0, 1.0, np.zeros(382)]),
            ):
                file_id = db.execute(
                    "INSERT INTO files(path, hash, filename) VALUES (?,?,?)",
                    (str(path), path.name, path.name),
                ).lastrowid
                chunk_id = db.execute(
                    "INSERT INTO chunks(file_id, chunk_index, text) VALUES (?,?,?)",
                    (file_id, 0, text),
                ).lastrowid
                db.execute(
                    "INSERT INTO vec_chunks(chunk_id, embedding) VALUES (?,?)",
                    (chunk_id, _serialize_vector(vector.astype(np.float32))),
                )
                db.execute(
                    "INSERT INTO chunks_fts(rowid, text) VALUES (?,?)",
                    (chunk_id, text),
                )
            db.commit()
            db.close()

            query_vector = np.r_[1.0, np.zeros(383)].astype(np.float32)[None, :]
            with patch.object(searcher, "embed", return_value=query_vector):
                results = searcher.search("revenue growth", db_path, top_k=2)

            unrelated_vector = np.r_[0.0, 0.0, 1.0, np.zeros(381)].astype(np.float32)[None, :]
            with patch.object(searcher, "embed", return_value=unrelated_vector):
                irrelevant = searcher.search("unrelated gibberish", db_path, top_k=2)

        self.assertEqual(results[0]["filename"], "quarterly.txt")
        self.assertIn("fts", results[0]["matched_by"])
        self.assertIn("semantic", results[0]["matched_by"])
        # Candidate retrieval no longer discards by an uncalibrated distance.
        # No-answer decisions belong to calibrated reranking, not this stage.
        self.assertTrue(irrelevant)
        self.assertTrue(all(result["semantic_distance"] == 1.0 for result in irrelevant))

    def test_semantic_query_quality_rejects_obvious_gibberish(self):
        self.assertFalse(searcher._semantic_query_is_usable("asdlkj qwrty zmxncb"))
        self.assertFalse(searcher._semantic_query_is_usable("zzzzzzzzzz"))
        self.assertTrue(searcher._semantic_query_is_usable("customer retention"))

    def test_legacy_l2_distance_is_compared_as_cosine_distance(self):
        self.assertAlmostEqual(searcher._as_cosine_distance(0.5, "l2"), 0.125)
        self.assertAlmostEqual(searcher._as_cosine_distance(0.125, "cosine"), 0.125)

    def test_search_runs_every_route_even_when_filename_fills_limit(self):
        db = MagicMock()
        filename = [_hit("/tmp/exact.txt")]
        fts = [_hit("/tmp/keyword.txt")]
        semantic = [_hit("/tmp/concept.txt")]

        with (
            patch.object(searcher, "_connect", return_value=db),
            patch.object(searcher, "_filename_search", return_value=filename) as names,
            patch.object(searcher, "_fts_search", return_value=fts) as keywords,
            patch.object(searcher, "_semantic_search", return_value=semantic) as vectors,
        ):
            results = searcher.search("exact", "/tmp/index.db", top_k=1)

        names.assert_called_once()
        keywords.assert_called_once()
        vectors.assert_called_once()
        db.close.assert_called_once_with()
        self.assertEqual(results[0]["path"], "/tmp/exact.txt")

    def test_rrf_rewards_multiple_routes(self):
        results = _fuse_ranked_results(
            [
                ("filename", [_hit("/tmp/name-only.txt")]),
                ("fts", [_hit("/tmp/multi.txt", "keyword evidence")]),
                ("semantic", [_hit("/tmp/multi.txt", "semantic evidence")]),
            ],
            top_k=5,
        )

        self.assertEqual(results[0]["path"], "/tmp/multi.txt")
        self.assertEqual(results[0]["matched_by"], ["fts", "semantic"])
        self.assertEqual(len(results[0]["evidence"]), 2)

    def test_rrf_keeps_strong_filename_priority(self):
        results = _fuse_ranked_results(
            [
                ("filename", [_hit("/tmp/exact.txt")]),
                ("semantic", [_hit("/tmp/concept.txt")]),
                ("fts", []),
            ],
            top_k=1,
        )

        self.assertEqual(results, [{
            "filename": "exact.txt",
            "path": "/tmp/exact.txt",
            "snippet": "",
            "score": 1.0,
            "matched_by": ["filename"],
            "evidence": [],
        }])

    def test_rrf_uses_content_snippet_and_normalizes_scores(self):
        results = _fuse_ranked_results(
            [
                ("filename", [_hit("/tmp/a.txt", "first chunk")]),
                ("fts", [_hit("/tmp/a.txt", "highlighted match")]),
                ("semantic", [_hit("/tmp/b.txt", "related passage")]),
            ],
            top_k=5,
        )

        self.assertEqual(results[0]["snippet"], "highlighted match")
        self.assertEqual(results[0]["score"], 1.0)
        self.assertTrue(all(0.0 < result["score"] <= 1.0 for result in results))


if __name__ == "__main__":
    unittest.main()
