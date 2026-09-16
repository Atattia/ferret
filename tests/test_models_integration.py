"""Optional real-model checks after downloading the documented ONNX presets."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from benchmarks.corpus import SCENARIOS
from core.maintenance import atomic_json, rebuild
from core.models import load_model
from core.searcher import search


MODEL = Path("models/qwen3-embedding-0.6b").resolve()


@unittest.skipUnless((MODEL / "ferret-model.json").exists(), "Download Qwen3 model first")
class ModelIntegrationTests(unittest.TestCase):
    def test_embedding_padding_and_query_cache(self):
        model = load_model(MODEL)
        texts = ["موظفون يتركون العمل بسبب الضغط", "A longer English explanation about employees leaving an exhausting workplace."]
        vectors = model.embed(texts)
        self.assertEqual(vectors.shape, (2, 1024))
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), [1., 1.], atol=1e-5)
        alone = model.embed([texts[0]])[0]
        np.testing.assert_array_equal(alone, vectors[0])
        first = model.embed(["pressure at work"], is_query=True)
        second = model.embed(["pressure at work"], is_query=True)
        np.testing.assert_array_equal(first, second)
        second[:] = 0
        self.assertGreater(np.linalg.norm(model.embed(["pressure at work"], is_query=True)), .99)

    def test_real_shadow_rebuild_activation_and_cross_language_search(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            documents = root / "documents"
            documents.mkdir()
            for topic in (2, 3):
                for extension, content in zip(("txt", "md"), SCENARIOS[topic][:2]):
                    (documents / f"record-{topic}.{extension}").write_text(content, encoding="utf-8")
            config_path = root / "settings.json"
            original = dict(db_path=str(root / "old.db"), indexed_folders=[str(documents)], ocr_languages="eng")
            atomic_json(config_path, original)
            report = rebuild(config_path, str(MODEL), str(root / "new.db"), activate=True)
            self.assertEqual(report["chunks"], 4)
            updated = json.loads(config_path.read_text())
            warnings = []
            results = search("refund for duplicate payment type:md", updated["db_path"],
                             model_path=updated["model_path"], mode="semantic", diagnostics=warnings)
            self.assertEqual(warnings, [])
            self.assertEqual(results[0]["filename"], "record-2.md")
            self.assertEqual(json.loads(Path(str(config_path) + ".previous").read_text()), original)
            # Resume the same complete shadow index without re-embedding.
            atomic_json(config_path, original)
            with patch("core.indexer.index_file", side_effect=AssertionError("unexpected reindex")):
                self.assertEqual(rebuild(config_path, str(MODEL), str(root / "new.db"))["chunks"], 4)


if __name__ == "__main__":
    unittest.main()
