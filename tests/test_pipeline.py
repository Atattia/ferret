"""
Ferret pipeline tests.

Run from project root:
    source venv/bin/activate
    python -m pytest tests/test_pipeline.py -v
    # or without pytest:
    python tests/test_pipeline.py
"""

import struct
import sqlite3
import tempfile
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import sqlite_vec
from core.indexer import init_db, index_file, index_folder, chunk_text, embed
from core.searcher import search
from core.hasher import hash_file

DB_PATH = None   # set per-test via fixture
MODEL = "~/ferret/models/bge-small-en"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_db() -> tuple[str, tempfile.TemporaryDirectory]:
    tmp = tempfile.TemporaryDirectory()
    db_path = str(Path(tmp.name) / "test.db")
    init_db(db_path)
    return db_path, tmp


def write(dir_: Path, name: str, text: str) -> Path:
    p = dir_ / name
    p.write_text(text, encoding="utf-8")
    return p


def db_counts(db_path: str) -> tuple[int, int, int]:
    db = sqlite3.connect(db_path)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    f = db.execute("SELECT count(*) FROM files").fetchone()[0]
    c = db.execute("SELECT count(*) FROM chunks").fetchone()[0]
    v = db.execute("SELECT count(*) FROM vec_chunks").fetchone()[0]
    db.close()
    return f, c, v


def result_paths(results: list[dict]) -> list[str]:
    return [Path(r["path"]).name for r in results]


# ---------------------------------------------------------------------------
# Chunking unit tests (no model needed)
# ---------------------------------------------------------------------------

def test_chunk_empty():
    assert chunk_text("") == []
    assert chunk_text("   ") == []

def test_chunk_short():
    # text shorter than chunk_size → single chunk
    text = " ".join(["word"] * 10)
    chunks = chunk_text(text, chunk_size=500)
    assert len(chunks) == 1

def test_chunk_overlap():
    # 1100 words, chunk_size=500, overlap=50 → step=450
    # chunk 0: words 0-499, chunk 1: words 450-949, chunk 2: words 900-1099
    text = " ".join([str(i) for i in range(1100)])
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    assert len(chunks) == 3
    # first word of chunk 1 should be word 450 (step=450)
    assert chunks[1].split()[0] == "450"

def test_chunk_single_word():
    chunks = chunk_text("hello", chunk_size=500)
    assert chunks == ["hello"]


# ---------------------------------------------------------------------------
# Embedding unit tests
# ---------------------------------------------------------------------------

def test_embed_shape():
    v = embed(["hello world"], MODEL)
    assert v.shape == (1, 384)

def test_embed_normalized():
    import numpy as np
    v = embed(["test sentence"], MODEL)
    norm = float(np.linalg.norm(v[0]))
    assert abs(norm - 1.0) < 1e-5, f"not normalized: norm={norm}"

def test_embed_batch_consistent():
    import numpy as np
    texts = ["the quick brown fox", "jumped over the lazy dog"]
    batch = embed(texts, MODEL)
    single0 = embed([texts[0]], MODEL)
    single1 = embed([texts[1]], MODEL)
    assert np.allclose(batch[0], single0[0], atol=1e-5)
    assert np.allclose(batch[1], single1[0], atol=1e-5)

def test_embed_empty():
    v = embed([], MODEL)
    assert v.shape == (0, 384)


# ---------------------------------------------------------------------------
# Indexing + search integration tests
# ---------------------------------------------------------------------------

def test_index_basic_txt():
    db_path, tmp = make_db()
    with tmp:
        d = Path(tmp.name)
        f = write(d, "notes.txt", "Python is a great programming language for data science.")
        index_file(f, db_path, MODEL)
        files, chunks, vecs = db_counts(db_path)
        assert files == 1
        assert chunks >= 1
        assert vecs == chunks

def test_index_skips_unchanged():
    db_path, tmp = make_db()
    with tmp:
        d = Path(tmp.name)
        f = write(d, "doc.txt", "Some content here.")
        index_file(f, db_path, MODEL)
        _, chunks_before, _ = db_counts(db_path)
        # Index same file again — should be skipped
        index_file(f, db_path, MODEL)
        _, chunks_after, _ = db_counts(db_path)
        assert chunks_before == chunks_after

def test_index_reindex_on_change():
    db_path, tmp = make_db()
    with tmp:
        d = Path(tmp.name)
        f = write(d, "doc.txt", "First version of the document.")
        index_file(f, db_path, MODEL)
        f.write_text("Completely different content about machine learning.", encoding="utf-8")
        index_file(f, db_path, MODEL)
        files, _, _ = db_counts(db_path)
        assert files == 1  # still one file record, not duplicated

def test_index_empty_file():
    db_path, tmp = make_db()
    with tmp:
        d = Path(tmp.name)
        f = write(d, "empty.txt", "")
        index_file(f, db_path, MODEL)
        files, chunks, _ = db_counts(db_path)
        assert files == 0  # empty file should not be stored
        assert chunks == 0

def test_index_whitespace_only():
    db_path, tmp = make_db()
    with tmp:
        d = Path(tmp.name)
        f = write(d, "blank.txt", "   \n\n\t  ")
        index_file(f, db_path, MODEL)
        files, _, _ = db_counts(db_path)
        assert files == 0

def test_index_missing_file():
    db_path, tmp = make_db()
    with tmp:
        # Should not raise; just print a warning
        index_file("/nonexistent/path/file.txt", db_path, MODEL)
        files, _, _ = db_counts(db_path)
        assert files == 0

def test_index_unsupported_extension():
    db_path, tmp = make_db()
    with tmp:
        d = Path(tmp.name)
        f = d / "image.png"
        f.write_bytes(b"\x89PNG\r\n")
        index_folder(str(d), db_path, workers=1, model_path=MODEL)
        files, _, _ = db_counts(db_path)
        assert files == 0

def test_index_special_chars_in_filename():
    db_path, tmp = make_db()
    with tmp:
        d = Path(tmp.name)
        f = write(d, "résumé (final) [v2].txt", "My professional experience includes software engineering.")
        index_file(f, db_path, MODEL)
        files, _, _ = db_counts(db_path)
        assert files == 1

def test_index_large_file():
    """~5000 words — should produce multiple chunks."""
    db_path, tmp = make_db()
    with tmp:
        d = Path(tmp.name)
        text = "The quick brown fox jumps over the lazy dog. " * 500  # ~5000 words
        f = write(d, "large.txt", text)
        index_file(f, db_path, MODEL)
        _, chunks, _ = db_counts(db_path)
        assert chunks > 1

def test_index_folder_multiple_files():
    db_path, tmp = make_db()
    with tmp:
        d = Path(tmp.name)
        write(d, "a.txt", "Document about astronomy and telescopes.")
        write(d, "b.txt", "Notes on cooking Italian pasta recipes.")
        write(d, "c.md", "Guide to learning the Rust programming language.")
        index_folder(str(d), db_path, workers=2, model_path=MODEL)
        files, _, _ = db_counts(db_path)
        assert files == 3


# ---------------------------------------------------------------------------
# Search tests
# ---------------------------------------------------------------------------

def test_search_empty_query():
    db_path, tmp = make_db()
    with tmp:
        assert search("", db_path, model_path=MODEL) == []
        assert search("   ", db_path, model_path=MODEL) == []

def test_search_no_results_empty_db():
    db_path, tmp = make_db()
    with tmp:
        results = search("machine learning", db_path, model_path=MODEL)
        assert results == []

def test_search_filename_match():
    db_path, tmp = make_db()
    with tmp:
        d = Path(tmp.name)
        write(d, "budget_2024.txt", "Q1 expenses were within projections.")
        index_folder(str(d), db_path, workers=1, model_path=MODEL)
        results = search("budget", db_path, top_k=5, model_path=MODEL)
        names = result_paths(results)
        assert "budget_2024.txt" in names
        # Filename match should be first
        assert names[0] == "budget_2024.txt"
        assert results[0]["score"] == 1.0

def test_search_filename_case_insensitive():
    db_path, tmp = make_db()
    with tmp:
        d = Path(tmp.name)
        write(d, "ProjectReport.txt", "Annual project summary.")
        index_folder(str(d), db_path, workers=1, model_path=MODEL)
        results = search("projectreport", db_path, top_k=5, model_path=MODEL)
        assert any("ProjectReport.txt" in r["path"] for r in results)

def test_search_semantic():
    db_path, tmp = make_db()
    with tmp:
        d = Path(tmp.name)
        write(d, "astro.txt", "The Milky Way galaxy contains billions of stars and several spiral arms.")
        write(d, "cooking.txt", "Sauté onions in olive oil until translucent, then add garlic.")
        index_folder(str(d), db_path, workers=1, model_path=MODEL)
        results = search("space and galaxies", db_path, top_k=2, model_path=MODEL)
        names = result_paths(results)
        assert "astro.txt" in names
        # astro should rank above cooking
        assert names.index("astro.txt") < names.index("cooking.txt") if "cooking.txt" in names else True

def test_search_filename_beats_semantic():
    """A filename match should appear before a semantically relevant but differently-named file."""
    db_path, tmp = make_db()
    with tmp:
        d = Path(tmp.name)
        # 'recipe.txt' is semantically irrelevant but its name matches
        write(d, "recipe.txt", "The photoelectric effect was discovered by Einstein.")
        # 'physics.txt' is semantically very relevant but name doesn't match
        write(d, "physics.txt", "Quantum mechanics describes particles at the subatomic level.")
        index_folder(str(d), db_path, workers=1, model_path=MODEL)
        results = search("recipe", db_path, top_k=5, model_path=MODEL)
        assert results[0]["filename"] == "recipe.txt"
        assert results[0]["score"] == 1.0

def test_search_top_k_respected():
    db_path, tmp = make_db()
    with tmp:
        d = Path(tmp.name)
        for i in range(10):
            write(d, f"doc_{i}.txt", f"Document number {i} about various interesting topics in science.")
        index_folder(str(d), db_path, workers=2, model_path=MODEL)
        results = search("science topics", db_path, top_k=3, model_path=MODEL)
        assert len(results) <= 3

def test_search_result_fields():
    db_path, tmp = make_db()
    with tmp:
        d = Path(tmp.name)
        write(d, "sample.txt", "Neural networks are a type of machine learning model.")
        index_folder(str(d), db_path, workers=1, model_path=MODEL)
        results = search("machine learning", db_path, top_k=1, model_path=MODEL)
        assert len(results) == 1
        r = results[0]
        assert "filename" in r
        assert "path" in r
        assert "snippet" in r
        assert "score" in r
        assert r["filename"] == "sample.txt"
        assert Path(r["path"]).exists()
        assert 0.0 <= r["score"] <= 1.0

def test_search_orphan_detection():
    db_path, tmp = make_db()
    with tmp:
        d = Path(tmp.name)
        f = write(d, "temp.txt", "Temporary document about solar energy and photovoltaics.")
        index_file(f, db_path, MODEL)
        f.unlink()  # delete the file
        results = search("solar energy", db_path, top_k=5, model_path=MODEL)
        # Deleted file should not appear in results
        assert not any("temp.txt" in r["path"] for r in results)


# ---------------------------------------------------------------------------
# Performance / stress
# ---------------------------------------------------------------------------

def test_index_speed_100_files():
    db_path, tmp = make_db()
    with tmp:
        d = Path(tmp.name)
        for i in range(100):
            write(d, f"file_{i:03d}.txt",
                  f"Document {i}: " + "lorem ipsum dolor sit amet. " * 20)
        t0 = time.time()
        index_folder(str(d), db_path, workers=4, model_path=MODEL)
        elapsed = time.time() - t0
        files, _, _ = db_counts(db_path)
        print(f"\n  100 files indexed in {elapsed:.1f}s ({elapsed/100*1000:.0f}ms/file)")
        assert files == 100

def test_search_latency():
    db_path, tmp = make_db()
    with tmp:
        d = Path(tmp.name)
        for i in range(50):
            write(d, f"doc_{i}.txt", f"Topic {i}: " + "interesting content about science. " * 10)
        index_folder(str(d), db_path, workers=4, model_path=MODEL)

        # Warm up (first search loads model into cache)
        search("science", db_path, top_k=5, model_path=MODEL)

        times = []
        for _ in range(5):
            t0 = time.time()
            search("interesting research findings", db_path, top_k=5, model_path=MODEL)
            times.append(time.time() - t0)

        avg_ms = sum(times) / len(times) * 1000
        print(f"\n  Search latency over 50 docs: avg={avg_ms:.0f}ms, max={max(times)*1000:.0f}ms")
        assert avg_ms < 2000, f"Search too slow: {avg_ms:.0f}ms avg"


# ---------------------------------------------------------------------------
# Runner (no pytest needed)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import traceback

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {fn.__name__}")
            traceback.print_exc()
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
