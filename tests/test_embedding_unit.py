"""Focused embedding tests using fake tokenizer and ONNX session objects."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import indexer


class FakeEncoding:
    def __init__(self, ids, attention_mask):
        self.ids = ids
        self.attention_mask = attention_mask


class FakeTokenizer:
    def __init__(self):
        self.texts = []
        self.padding_kwargs = None
        self.truncation_kwargs = None

    def enable_padding(self, **kwargs):
        self.padding_kwargs = kwargs

    def enable_truncation(self, **kwargs):
        self.truncation_kwargs = kwargs

    def encode_batch(self, texts):
        self.texts = list(texts)
        # Deliberately use different sequence lengths: embed must consume the
        # batch's actual (dynamically padded) shape, not assume length 512.
        return [
            FakeEncoding([1, 2], [1, 1]),
            FakeEncoding([3, 4], [1, 1]),
        ][: len(texts)]


class FakeSession:
    def __init__(self):
        self.inputs = []

    def run(self, _outputs, inputs):
        self.inputs.append(inputs)
        batch = inputs["input_ids"].shape[0]
        # CLS vectors differ from the masked mean, making pooling behavior
        # directly observable.
        output = np.array(
            [
                [[3.0, 4.0], [100.0, 0.0]],
                [[0.0, 5.0], [0.0, 100.0]],
            ][:batch],
            dtype=np.float32,
        )
        return [output]


def _fake_session():
    tokenizer = FakeTokenizer()
    session = FakeSession()
    return tokenizer, session


class EmbeddingTests(unittest.TestCase):
    def test_embed_uses_cls_pooling_and_normalizes(self):
        tokenizer, session = _fake_session()
        with patch.object(indexer, "_get_session", return_value=(tokenizer, session)):
            vectors = indexer.embed(["document", "another"])

        self.assertEqual(vectors.shape, (2, 2))
        np.testing.assert_allclose(vectors[0], [0.6, 0.8])
        np.testing.assert_allclose(vectors[1], [0.0, 1.0])
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), [1.0, 1.0])

    def test_query_embedding_adds_bge_instruction(self):
        tokenizer, session = _fake_session()
        with patch.object(indexer, "_get_session", return_value=(tokenizer, session)):
            indexer.embed_queries(["find my report"])

        self.assertEqual(tokenizer.texts, [
            indexer.BGE_QUERY_INSTRUCTION + "find my report"
        ])

    def test_legacy_embed_is_document_embedding(self):
        tokenizer, session = _fake_session()
        with patch.object(indexer, "_get_session", return_value=(tokenizer, session)):
            indexer.embed(["find my report"])

        self.assertEqual(tokenizer.texts, ["find my report"])

    def test_empty_query_helpers_have_expected_shapes(self):
        with patch.object(indexer, "_get_session", return_value=_fake_session()):
            self.assertEqual(indexer.embed_queries([]).shape, (0, 384))

    def test_session_cache_is_keyed_by_model_path(self):
        calls = []
        tokenizers = []

        class FakeTokenizerFactory:
            @staticmethod
            def from_file(path):
                tokenizer = FakeTokenizer()
                calls.append(path)
                tokenizers.append(tokenizer)
                return tokenizer

        class FakeTokenizerModule:
            Tokenizer = FakeTokenizerFactory

        class FakeOrt:
            class SessionOptions:
                def __init__(self):
                    self.intra_op_num_threads = None

            class InferenceSession:
                def __init__(self, path, sess_options):
                    self.path = path

        with patch.dict(sys.modules, {
            "tokenizers": FakeTokenizerModule,
            "onnxruntime": FakeOrt,
        }), patch.object(indexer, "_session", None), patch.object(
            indexer, "_session_model_path", None
        ):
            import tempfile
            with tempfile.TemporaryDirectory() as tmp:
                first = Path(tmp) / "model-a"
                second = Path(tmp) / "model-b"
                indexer._get_session(str(first))
                indexer._get_session(str(first))
                indexer._get_session(str(second))

        self.assertEqual(len(calls), 2)
        self.assertTrue(calls[0].endswith("model-a/tokenizer.json"))
        self.assertTrue(calls[1].endswith("model-b/tokenizer.json"))
        # The model limit is retained, but padding is dynamic per batch.
        self.assertEqual(tokenizers[0].padding_kwargs, {"pad_token": "[PAD]"})
        self.assertEqual(tokenizers[0].truncation_kwargs, {"max_length": 512})

if __name__ == "__main__":
    unittest.main()
