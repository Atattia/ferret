"""Latency contracts: opt-in reranking and query-priority inference scheduling."""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np

from core import indexer, searcher
from core.models import InferenceGate, ModelSpec, OnnxModel, configured_reranker


class SearchLatencyTests(unittest.TestCase):
    def test_reranking_requires_explicit_opt_in(self):
        self.assertIsNone(configured_reranker({"reranker_path": "model"}))
        self.assertIsNone(configured_reranker({"reranker_path": "model", "reranking_enabled": False}))
        self.assertEqual(configured_reranker({"reranker_path": "model", "reranking_enabled": True}), "model")

    def test_query_preempts_document_between_embedding_batches(self):
        model = OnnxModel.__new__(OnnxModel)
        model.spec = ModelSpec(dimensions=2, pooling="last", query_prefix="")
        model.lock = threading.RLock()
        model.inference_gate = InferenceGate()
        model.query_cache = OrderedDict()
        started, release = threading.Event(), threading.Event()
        calls = []

        def run(items):
            calls.extend(items)
            if items == ["p1"]:
                started.set()
                if not release.wait(5):
                    raise RuntimeError("test timed out")
            return np.ones((1, 1, 2)), np.ones((1, 1), dtype=np.int64)

        model._run = run
        with ThreadPoolExecutor(max_workers=2) as pool:
            indexing = pool.submit(model.embed, ["p1", "p2", "p3"])
            try:
                self.assertTrue(started.wait(2))
                query = pool.submit(model.embed, ["q"], True)
                with model.inference_gate.condition:
                    self.assertTrue(model.inference_gate.condition.wait_for(
                        lambda: model.inference_gate.queries == 1, timeout=2))
            finally:
                release.set()
            query.result(timeout=5)
            indexing.result(timeout=5)
        self.assertEqual(calls, ["p1", "q", "p2", "p3"])

    def test_gate_recovers_after_failed_inference(self):
        gate = InferenceGate()
        with self.assertRaises(ValueError):
            with gate.slot():
                raise ValueError("inference failed")
        with gate.slot(query=True):
            self.assertTrue(gate.busy)
        self.assertFalse(gate.busy)

    def test_cancelled_search_does_not_start_semantic_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "index.db")
            indexer.init_db(db)
            with patch.object(searcher, "_semantic_search") as semantic:
                self.assertEqual(searcher.search("find a file", db, cancelled=lambda: True), [])
                semantic.assert_not_called()

    def test_keyword_mode_never_runs_reranker(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "index.db")
            indexer.init_db(db)
            candidate = dict(path="file", filename="file", snippet="a passage", score=1., matched_by=["fts"])
            with patch.object(searcher, "_fuse_ranked_results", return_value=[candidate]), \
                 patch.object(searcher, "_rerank") as rerank:
                searcher.search("find a file", db, mode="keyword", reranker_path="model")
                rerank.assert_not_called()
