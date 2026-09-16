"""Dependency-free tests for token-aware, structure-preserving chunking."""

from pathlib import Path
import re
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.indexer import (
    chunk_content_hash,
    chunk_document,
    chunk_text,
    chunk_text_by_tokens,
)


class FakeEncoding:
    def __init__(self, ids):
        self.ids = ids


class FakeTokenizer:
    """Approximate a BGE tokenizer with two special tokens per encoding."""

    def __init__(self, truncated_at=None):
        self.truncation = (
            {
                "max_length": truncated_at,
                "stride": 0,
                "strategy": "longest_first",
                "direction": "right",
            }
            if truncated_at is not None
            else None
        )
        self.restore_calls = []

    def encode(self, text):
        # Treat words, punctuation, and each non-ASCII character as a token so
        # tests exercise encodings instead of duplicating whitespace counts.
        content = re.findall(r"[A-Za-z0-9_]+|[^\x00-\x7f]|[^\w\s]", text)
        ids = [101] + list(range(len(content))) + [102]
        if self.truncation is not None:
            ids = ids[: self.truncation["max_length"]]
        return FakeEncoding(ids)

    def no_truncation(self):
        self.truncation = None

    def enable_truncation(self, **kwargs):
        self.restore_calls.append(kwargs)
        self.truncation = dict(kwargs)


class TokenAwareChunkingTests(unittest.TestCase):
    def test_document_chunks_preserve_page_line_and_heading_metadata(self):
        tokenizer = FakeTokenizer()
        records = chunk_document(
            "# First section\nalpha beta\f# Second section\ngamma delta",
            tokenizer,
            max_tokens=20,
            overlap_tokens=0,
        )

        self.assertEqual([record.page_number for record in records], [1, 2])
        self.assertEqual([record.start_line for record in records], [1, 1])
        self.assertEqual([record.end_line for record in records], [2, 2])
        self.assertEqual(
            [record.heading for record in records],
            ["First section", "Second section"],
        )
        self.assertTrue(all(record.content_hash for record in records))

    def test_legacy_word_chunker_remains_compatible(self):
        self.assertEqual(
            chunk_text("one two three four five", chunk_size=3, overlap=1),
            ["one two three", "three four five", "five"],
        )

    def test_chunks_never_exceed_encoded_token_limit(self):
        tokenizer = FakeTokenizer()
        text = " ".join(f"word{i}" for i in range(30))

        chunks = chunk_text_by_tokens(
            text, tokenizer, max_tokens=10, overlap_tokens=4
        )

        self.assertGreater(len(chunks), 1)
        self.assertTrue(
            all(len(tokenizer.encode(chunk).ids) <= 10 for chunk in chunks)
        )
        self.assertIn("word0", chunks[0])
        self.assertIn("word29", chunks[-1])

    def test_preserves_paragraph_and_line_boundaries_when_they_fit(self):
        tokenizer = FakeTokenizer()
        text = "First line\nSecond line\n\nAnother paragraph"

        chunks = chunk_text_by_tokens(text, tokenizer, max_tokens=30, overlap_tokens=0)

        self.assertEqual(chunks, [text])

    def test_prefers_paragraph_boundary_when_combined_text_is_too_large(self):
        tokenizer = FakeTokenizer()
        text = "Alpha beta\n\nGamma delta"

        chunks = chunk_text_by_tokens(text, tokenizer, max_tokens=5, overlap_tokens=0)

        self.assertEqual(chunks, ["Alpha beta", "Gamma delta"])

    def test_overlong_unbroken_text_uses_safe_character_fallback(self):
        tokenizer = FakeTokenizer()
        text = "界" * 23

        chunks = chunk_text_by_tokens(text, tokenizer, max_tokens=8, overlap_tokens=0)

        self.assertEqual("".join(chunks), text)
        self.assertTrue(all(len(tokenizer.encode(chunk).ids) <= 8 for chunk in chunks))

    def test_temporarily_disables_and_restores_global_truncation(self):
        tokenizer = FakeTokenizer(truncated_at=12)
        original = dict(tokenizer.truncation)
        text = " ".join(f"item{i}" for i in range(40))

        chunks = chunk_text_by_tokens(text, tokenizer, max_tokens=12, overlap_tokens=0)

        self.assertGreater(len(chunks), 1)
        self.assertEqual(tokenizer.truncation, original)
        self.assertEqual(tokenizer.restore_calls, [original])

    def test_chunk_hash_is_stable_position_independent_sha256(self):
        text = "same chunk text"

        first = chunk_content_hash(text)
        second = chunk_content_hash(text)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)
        self.assertNotEqual(first, chunk_content_hash(text + "!"))

    def test_rejects_overlap_that_cannot_make_forward_progress(self):
        tokenizer = FakeTokenizer()
        with self.assertRaises(ValueError):
            chunk_text_by_tokens("text", tokenizer, max_tokens=8, overlap_tokens=8)


if __name__ == "__main__":
    unittest.main()
