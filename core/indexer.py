import json
import sqlite3
import struct
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import sqlite_vec

from core.extractor import extract
from core.hasher import hash_file


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks by word count."""
    if not text.strip():
        return []
    words = text.split()
    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

_session = None  # module-level cache (per-process)


def _get_session(model_path: str):
    global _session
    if _session is None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        model_dir = Path(model_path).expanduser()
        onnx_path = model_dir / "onnx" / "model.onnx"
        tokenizer_path = model_dir / "tokenizer.json"

        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        tokenizer.enable_padding(pad_token="[PAD]", length=512)
        tokenizer.enable_truncation(max_length=512)

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        session = ort.InferenceSession(str(onnx_path), sess_options=sess_options)

        _session = (tokenizer, session)
    return _session


def embed(texts: list[str], model_path: str = "~/ferret/models/bge-small-en") -> np.ndarray:
    """Embed a list of texts using bge-small-en ONNX model. Returns (N, 384) normalized float32 array."""
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)

    tokenizer, session = _get_session(model_path)
    encodings = tokenizer.encode_batch(texts)

    input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
    attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
    token_type_ids = np.zeros_like(input_ids, dtype=np.int64)

    outputs = session.run(None, {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    })

    # Mean pool over token dimension using attention mask
    token_embeddings = outputs[0]  # (batch, seq_len, hidden)
    mask = attention_mask[:, :, np.newaxis].astype(np.float32)
    summed = (token_embeddings * mask).sum(axis=1)
    counts = mask.sum(axis=1).clip(min=1e-9)
    pooled = summed / counts

    # L2 normalize
    norms = np.linalg.norm(pooled, axis=1, keepdims=True).clip(min=1e-9)
    return (pooled / norms).astype(np.float32)


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def _connect(db_path: str) -> sqlite3.Connection:
    db = sqlite3.connect(str(Path(db_path).expanduser()))
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    db.execute("PRAGMA journal_mode=WAL")
    return db


def init_db(db_path: str) -> None:
    """Create the SQLite DB with sqlite-vec virtual table and files metadata table."""
    db = _connect(db_path)
    db.executescript("""
        CREATE TABLE IF NOT EXISTS files (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            path     TEXT UNIQUE NOT NULL,
            hash     TEXT NOT NULL,
            filename TEXT NOT NULL,
            status   TEXT NOT NULL DEFAULT 'indexed'
        );

        CREATE TABLE IF NOT EXISTS chunks (
            id      INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
            chunk_index INTEGER NOT NULL,
            text    TEXT NOT NULL
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
            chunk_id INTEGER PRIMARY KEY,
            embedding FLOAT[384]
        );
    """)
    db.commit()
    db.close()
    print(f"[indexer] DB initialized at {db_path}")


# ---------------------------------------------------------------------------
# Per-file indexing
# ---------------------------------------------------------------------------

def _serialize_vector(v: np.ndarray) -> bytes:
    return struct.pack(f"{len(v)}f", *v.tolist())


def _delete_file_data(db: sqlite3.Connection, file_id: int) -> None:
    """Remove all chunk and vector data for a file."""
    chunk_ids = [r[0] for r in db.execute(
        "SELECT id FROM chunks WHERE file_id = ?", (file_id,)
    )]
    if chunk_ids:
        placeholders = ",".join("?" * len(chunk_ids))
        db.execute(f"DELETE FROM vec_chunks WHERE chunk_id IN ({placeholders})", chunk_ids)
        db.execute(f"DELETE FROM chunks WHERE id IN ({placeholders})", chunk_ids)


def index_file(path: str | Path, db_path: str, model_path: str = "~/ferret/models/bge-small-en") -> None:
    """Full pipeline: extract → chunk → embed → store. Skips if hash unchanged."""
    path = Path(path).resolve()
    if not path.exists():
        print(f"[indexer] File not found, skipping: {path}")
        return

    current_hash = hash_file(path)
    if not current_hash:
        return

    db = _connect(db_path)
    try:
        row = db.execute(
            "SELECT id, hash FROM files WHERE path = ?", (str(path),)
        ).fetchone()

        if row and row[1] == current_hash:
            print(f"[indexer] Unchanged, skipping: {path.name}")
            return

        text = extract(path)
        if not text.strip():
            print(f"[indexer] No text extracted from: {path.name}")
            if row:
                db.execute("UPDATE files SET status='indexed', hash=? WHERE id=?", (current_hash, row[0]))
                db.commit()
            return

        chunks = chunk_text(text)
        if not chunks:
            return

        vectors = embed(chunks, model_path)

        if row:
            file_id = row[0]
            _delete_file_data(db, file_id)
            db.execute(
                "UPDATE files SET hash=?, filename=?, status='indexed' WHERE id=?",
                (current_hash, path.name, file_id),
            )
        else:
            cursor = db.execute(
                "INSERT INTO files (path, hash, filename, status) VALUES (?,?,?,'indexed')",
                (str(path), current_hash, path.name),
            )
            file_id = cursor.lastrowid

        for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
            cursor = db.execute(
                "INSERT INTO chunks (file_id, chunk_index, text) VALUES (?,?,?)",
                (file_id, i, chunk),
            )
            chunk_id = cursor.lastrowid
            db.execute(
                "INSERT INTO vec_chunks (chunk_id, embedding) VALUES (?,?)",
                (chunk_id, _serialize_vector(vec)),
            )

        db.commit()
        print(f"[indexer] Indexed {path.name}: {len(chunks)} chunks")
    except Exception as e:
        print(f"[indexer] Error indexing {path}: {e}")
        db.rollback()
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Folder indexing
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


def _index_file_worker(args: tuple) -> None:
    path, db_path, model_path = args
    index_file(path, db_path, model_path)


def _load_config() -> dict:
    config_path = Path(__file__).parent.parent / "config" / "settings.json"
    try:
        with open(config_path) as f:
            return json.load(f)
    except Exception:
        return {}


def reset_file_hashes(db_path: str) -> int:
    """
    Clear the stored hash for every indexed file so the next reindex
    re-extracts all content regardless of whether the file changed on disk.
    Returns the number of files reset.
    """
    db = _connect(db_path)
    cursor = db.execute("UPDATE files SET hash = '' WHERE status = 'indexed'")
    count = cursor.rowcount
    db.commit()
    db.close()
    print(f"[indexer] Reset hashes for {count} file(s) — next reindex will re-extract all")
    return count


def index_folder(
    folder: str | Path,
    db_path: str,
    workers: int = 4,
    model_path: str = "~/ferret/models/bge-small-en",
    exclude_patterns: list[str] | None = None,
) -> None:
    """Index all supported files in a folder using a multiprocessing pool."""
    folder = Path(folder).resolve()
    if not folder.is_dir():
        print(f"[indexer] Not a directory: {folder}")
        return

    config = _load_config()
    if exclude_patterns is None:
        exclude_patterns = config.get("exclude_patterns", [])

    files = []
    for path in folder.rglob("*"):
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if any(pat in path.parts for pat in exclude_patterns):
            continue
        files.append(path)

    print(f"[indexer] Found {len(files)} files to index in {folder}")

    args = [(str(f), db_path, model_path) for f in files]
    with Pool(processes=workers) as pool:
        pool.map(_index_file_worker, args)

    print(f"[indexer] Folder indexing complete: {folder}")
