import hashlib
import json
import re
import sqlite3
import struct
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import sqlite_vec

from core.extractor import extract, extraction_warnings
from core.hasher import hash_file
from core.language import normalize, lexical_text
from core.models import has_manifest, model_spec, load_model, check_index_model


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


BGE_MAX_TOKENS = 512


@dataclass(frozen=True)
class DocumentChunk:
    text: str
    content_hash: str
    page_number: int
    start_line: int
    end_line: int
    heading: str | None = None


def chunk_content_hash(text: str) -> str:
    """Return a stable content identifier for a chunk.

    Hashing the exact UTF-8 text makes the value independent of chunk position
    and file identity, which allows a later indexer migration to reuse an
    unchanged chunk's embedding without changing today's database workflow.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _token_count(tokenizer, text: str) -> int:
    """Count the ids produced by a tokenizers-compatible tokenizer."""
    encoding = tokenizer.encode(text)
    return len(encoding.ids)


def _split_oversized_fragment(
    fragment: str, tokenizer, max_tokens: int
) -> list[str]:
    """Split a whitespace-free fragment, as a last resort, on characters."""
    pieces = []
    remaining = fragment
    while remaining:
        low, high = 1, len(remaining)
        best = 0
        while low <= high:
            middle = (low + high) // 2
            if _token_count(tokenizer, remaining[:middle]) <= max_tokens:
                best = middle
                low = middle + 1
            else:
                high = middle - 1
        if best == 0:
            raise ValueError("max_tokens is too small for even one character")
        pieces.append(remaining[:best])
        remaining = remaining[best:]
    return pieces


def _split_oversized_line(line: str, tokenizer, max_tokens: int) -> list[str]:
    """Split a line at word boundaries, falling back to character boundaries."""
    words = re.findall(r"\S+", line)
    pieces = []
    current = ""
    for word in words:
        candidate = f"{current} {word}" if current else word
        if _token_count(tokenizer, candidate) <= max_tokens:
            current = candidate
            continue

        if current:
            pieces.append(current)
            current = ""
        if _token_count(tokenizer, word) <= max_tokens:
            current = word
            continue

        word_pieces = _split_oversized_fragment(word, tokenizer, max_tokens)
        pieces.extend(word_pieces[:-1])
        current = word_pieces[-1]

    if current:
        pieces.append(current)
    return pieces


def _structural_units(text: str, tokenizer, max_tokens: int) -> list[tuple[str, str]]:
    """Return bounded text units and the preferred separator before each one."""
    units: list[tuple[str, str]] = []
    paragraphs = re.split(r"\n[ \t]*\n+", text.strip())
    for paragraph_index, paragraph in enumerate(paragraphs):
        lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
        for line_index, line in enumerate(lines):
            if paragraph_index and line_index == 0:
                separator = "\n\n"
            elif line_index:
                separator = "\n"
            else:
                separator = ""

            if _token_count(tokenizer, line) <= max_tokens:
                units.append((separator, line))
                continue

            pieces = _split_oversized_line(line, tokenizer, max_tokens)
            for piece_index, piece in enumerate(pieces):
                units.append((separator if piece_index == 0 else " ", piece))
    return units


def _overlap_suffix(text: str, tokenizer, overlap_tokens: int) -> str:
    """Choose the longest bounded suffix beginning at a natural boundary."""
    if not text or overlap_tokens == 0:
        return ""

    # Prefer a useful structural suffix. Word starts are included so a single
    # long line can still overlap without cutting through a word.
    starts = {0}
    starts.update(match.end() for match in re.finditer(r"\n[ \t]*\n+|\n|\s+", text))
    for start in sorted(starts):
        suffix = text[start:].strip()
        if suffix and _token_count(tokenizer, suffix) <= overlap_tokens:
            return suffix
    return ""


def chunk_text_by_tokens(
    text: str,
    tokenizer,
    max_tokens: int = BGE_MAX_TOKENS,
    overlap_tokens: int = 50,
) -> list[str]:
    """Split text into structure-preserving, token-bounded passages.

    ``tokenizer`` only needs an ``encode(text)`` method returning an object
    with an ``ids`` sequence. Paragraph and line breaks are retained whenever
    they fit. Oversized lines split at words, with character splitting reserved
    for unusually long individual tokens/strings.

    If the tokenizer currently has global truncation enabled (as Ferret's BGE
    tokenizer does), it is temporarily disabled so the limit is measured
    rather than silently truncated, then restored before returning.
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if overlap_tokens < 0 or overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be non-negative and less than max_tokens")
    if not text.strip():
        return []

    truncation = getattr(tokenizer, "truncation", None)
    no_truncation = getattr(tokenizer, "no_truncation", None)
    if truncation is not None and callable(no_truncation):
        no_truncation()

    try:
        units = _structural_units(text, tokenizer, max_tokens)
        chunks: list[str] = []
        current = ""
        for separator, unit in units:
            candidate = current + (separator if current else "") + unit
            if _token_count(tokenizer, candidate) <= max_tokens:
                current = candidate
                continue

            if current:
                chunks.append(current)
            carry = _overlap_suffix(current, tokenizer, overlap_tokens)
            candidate = carry + (separator if carry else "") + unit
            current = candidate if _token_count(tokenizer, candidate) <= max_tokens else unit

        if current:
            chunks.append(current)

        # Keep the model-limit invariant explicit even if a custom tokenizer
        # has surprising tokenization around joined boundaries.
        if any(_token_count(tokenizer, chunk) > max_tokens for chunk in chunks):
            raise RuntimeError("token-aware chunking produced an oversized passage")
        return chunks
    finally:
        if truncation is not None:
            tokenizer.enable_truncation(**truncation)


def _chunk_line_range(page_text: str, chunk: str) -> tuple[int, int]:
    """Locate a normalized chunk within its source page, best effort."""
    source_lines = page_text.splitlines() or [page_text]
    chunk_lines = [line.strip() for line in chunk.splitlines() if line.strip()]
    if not chunk_lines:
        return 1, 1

    def locate(fragment: str, start: int = 0) -> int:
        for index in range(start, len(source_lines)):
            if fragment in source_lines[index].strip():
                return index
        return start

    start = locate(chunk_lines[0])
    end = locate(chunk_lines[-1], start)
    return start + 1, end + 1


def _heading_before_line(page_text: str, line_number: int) -> str | None:
    """Return the nearest Markdown-style heading preceding a chunk."""
    heading = None
    for line in page_text.splitlines()[:line_number]:
        match = re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$", line)
        if match:
            heading = match.group(1)
    return heading


def chunk_document(
    text: str,
    tokenizer,
    max_tokens: int = BGE_MAX_TOKENS,
    overlap_tokens: int = 50,
) -> list[DocumentChunk]:
    """Build token-safe chunks with PDF page and text line metadata."""
    records = []
    for page_number, page_text in enumerate(text.split("\f"), start=1):
        if not page_text.strip():
            continue
        for chunk in chunk_text_by_tokens(
            page_text,
            tokenizer,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
        ):
            start_line, end_line = _chunk_line_range(page_text, chunk)
            records.append(DocumentChunk(
                text=chunk,
                content_hash=chunk_content_hash(chunk),
                page_number=page_number,
                start_line=start_line,
                end_line=end_line,
                heading=_heading_before_line(page_text, start_line),
            ))
    return records


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

_session = None  # module-level cache (per-process)
_session_model_path = None

# BGE's asymmetric retrieval instruction is used for queries only.  Passage
# embeddings must remain unprefixed, as recommended by the BGE model card.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
EMBEDDING_PIPELINE_VERSION = "bge-cls-query-instruction-token-chunks-v2"


def _normalized_model_path(model_path: str | Path) -> str:
    """Return a stable cache key for a model path."""
    return str(Path(model_path).expanduser().resolve())


def _get_session(model_path: str):
    global _session, _session_model_path
    cache_key = _normalized_model_path(model_path)
    if _session is None or _session_model_path != cache_key:
        import onnxruntime as ort
        if hasattr(ort, "disable_telemetry_events"):
            ort.disable_telemetry_events()
        from tokenizers import Tokenizer

        model_dir = Path(cache_key)
        onnx_path = model_dir / "onnx" / "model.onnx"
        tokenizer_path = model_dir / "tokenizer.json"

        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        # Without a fixed length, tokenizers pads only to the longest item in
        # the current batch.  Truncation remains bounded by the model limit.
        tokenizer.enable_padding(pad_token="[PAD]")
        tokenizer.enable_truncation(max_length=512)

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        session = ort.InferenceSession(str(onnx_path), sess_options=sess_options)

        _session = (tokenizer, session)
        _session_model_path = cache_key
    return _session


def embed(
    texts: list[str],
    model_path: str = "~/ferret/models/bge-small-en",
    *,
    is_query: bool = False,
) -> np.ndarray:
    """Embed texts with BGE and return an ``(N, 384)`` normalized array.

    The historical ``embed(texts, model_path)`` call remains document
    embedding (``is_query=False``).  Query embeddings use BGE's retrieval
    instruction via ``is_query=True`` or the clearer :func:`embed_queries`
    helper.
    """
    if has_manifest(model_path):
        return load_model(model_path).embed(texts, is_query=is_query)
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)

    tokenizer, session = _get_session(model_path)
    encoded_texts = texts
    if is_query:
        encoded_texts = [BGE_QUERY_INSTRUCTION + text for text in texts]
    encodings = tokenizer.encode_batch(encoded_texts)

    input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
    attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
    token_type_ids = np.zeros_like(input_ids, dtype=np.int64)

    outputs = session.run(None, {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    })

    # BGE uses the CLS representation, rather than mean pooling the token
    # sequence.  Keep the mask in the model inputs, but do not pool padding.
    token_embeddings = outputs[0]  # (batch, seq_len, hidden)
    pooled = token_embeddings[:, 0, :]

    # L2 normalize
    norms = np.linalg.norm(pooled, axis=1, keepdims=True).clip(min=1e-9)
    return (pooled / norms).astype(np.float32)


def embed_queries(
    texts: list[str], model_path: str = "~/ferret/models/bge-small-en"
) -> np.ndarray:
    """Embed search queries using BGE's query-side retrieval instruction."""
    return embed(texts, model_path, is_query=True)


def embed_query(
    text: str, model_path: str = "~/ferret/models/bge-small-en"
) -> np.ndarray:
    """Embed one search query and return its normalized one-dimensional vector."""
    return embed_queries([text], model_path)[0]


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


def init_db(db_path: str, model_path: str | None = None) -> bool:
    """Initialize storage and return whether existing files need re-indexing."""
    spec = model_spec(model_path) if model_path else model_spec("")
    dimensions = spec.dimensions
    db = _connect(db_path)
    existing = db.execute("SELECT 1 FROM sqlite_master WHERE name='index_metadata'").fetchone()
    if existing and model_path:
        try:
            check_index_model(db, model_path)
        except Exception:
            db.close()
            raise
    db.executescript(f"""
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
            text    TEXT NOT NULL,
            content_hash TEXT NOT NULL DEFAULT '',
            page_number INTEGER NOT NULL DEFAULT 1,
            start_line INTEGER NOT NULL DEFAULT 1,
            end_line INTEGER NOT NULL DEFAULT 1,
            heading TEXT
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
            chunk_id INTEGER PRIMARY KEY,
            embedding FLOAT[{dimensions}] distance_metric=cosine
        );

        CREATE TABLE IF NOT EXISTS document_vectors (
            file_id INTEGER PRIMARY KEY REFERENCES files(id) ON DELETE CASCADE,
            embedding BLOB NOT NULL
        );
        CREATE TRIGGER IF NOT EXISTS document_vectors_delete AFTER DELETE ON files BEGIN
            DELETE FROM document_vectors WHERE file_id=old.id;
        END;

        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            text,
            content='chunks',
            content_rowid='id'
        );

        CREATE TABLE IF NOT EXISTS index_metadata (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS chunks_file_position ON chunks(file_id, chunk_index);
        CREATE INDEX IF NOT EXISTS files_status ON files(status);

        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_ar_fts USING fts5(text);
        CREATE TRIGGER IF NOT EXISTS chunks_ar_delete AFTER DELETE ON chunks BEGIN
            DELETE FROM chunks_ar_fts WHERE rowid=old.id;
        END;
    """)
    chunk_columns = {
        row[1] for row in db.execute("PRAGMA table_info(chunks)").fetchall()
    }
    file_columns = {row[1] for row in db.execute("PRAGMA table_info(files)")}
    if "extraction_warning" not in file_columns:
        db.execute("ALTER TABLE files ADD COLUMN extraction_warning TEXT NOT NULL DEFAULT ''")
    if "content_hash" not in chunk_columns:
        db.execute(
            "ALTER TABLE chunks ADD COLUMN content_hash TEXT NOT NULL DEFAULT ''"
        )
    for column, definition in (
        ("page_number", "INTEGER NOT NULL DEFAULT 1"),
        ("start_line", "INTEGER NOT NULL DEFAULT 1"),
        ("end_line", "INTEGER NOT NULL DEFAULT 1"),
        ("heading", "TEXT"),
    ):
        if column not in chunk_columns:
            db.execute(f"ALTER TABLE chunks ADD COLUMN {column} {definition}")
    stored_version = db.execute(
        "SELECT value FROM index_metadata WHERE key='embedding_pipeline_version'"
    ).fetchone()
    chunk_count = db.execute("SELECT count(*) FROM chunks").fetchone()[0]
    pipeline_changed = bool(
        chunk_count and (
            stored_version is None or stored_version[0] != EMBEDDING_PIPELINE_VERSION
        )
    )
    if pipeline_changed:
        # Keep cached rows available for replacement, but exclude incompatible
        # embeddings from every search route until each source file is rebuilt.
        db.execute(
            """UPDATE files SET hash='', status='stale'
               WHERE id IN (SELECT DISTINCT file_id FROM chunks)"""
        )
        print("[indexer] Embedding pipeline changed; scheduled a full re-index")
    stale_count = db.execute(
        "SELECT count(*) FROM files WHERE status='stale'"
    ).fetchone()[0]
    # Retry on later launches if a prior rebuild could not finish (for example,
    # because its model or source folder was temporarily unavailable).
    requires_reindex = bool(stale_count)
    db.execute(
        """
        INSERT INTO index_metadata(key, value)
        VALUES ('embedding_pipeline_version', ?)
        ON CONFLICT(key) DO UPDATE SET value=excluded.value
        """,
        (EMBEDDING_PIPELINE_VERSION,),
    )
    if model_path:
        db.execute("INSERT OR IGNORE INTO index_metadata(key,value) VALUES ('model_fingerprint',?)",
                   (spec.fingerprint,))
        db.execute("INSERT OR IGNORE INTO index_metadata(key,value) VALUES ('model_spec',?)",
                   (json.dumps(spec.__dict__),))
    for chunk_id, text in db.execute(
        "SELECT id,text FROM chunks WHERE id NOT IN (SELECT rowid FROM chunks_ar_fts)"
    ).fetchall():
        db.execute("INSERT INTO chunks_ar_fts(rowid,text) VALUES (?,?)", (chunk_id, lexical_text(text)))
    for (file_id,) in db.execute(
        "SELECT DISTINCT file_id FROM chunks WHERE file_id NOT IN (SELECT file_id FROM document_vectors)"
    ).fetchall():
        vector_rows = db.execute("SELECT vc.embedding FROM vec_chunks vc JOIN chunks c ON c.id=vc.chunk_id WHERE c.file_id=?",
                                 (file_id,)).fetchall()
        if vector_rows:
            pooled = np.mean([np.frombuffer(blob, dtype=np.float32) for (blob,) in vector_rows], axis=0)
            pooled /= max(float(np.linalg.norm(pooled)), 1e-9)
            db.execute("INSERT INTO document_vectors(file_id,embedding) VALUES (?,?)",
                       (file_id, _serialize_vector(pooled.astype(np.float32))))
    db.commit()
    db.close()
    print(f"[indexer] DB initialized at {db_path}")
    return requires_reindex


# ---------------------------------------------------------------------------
# Per-file indexing
# ---------------------------------------------------------------------------

def _serialize_vector(v: np.ndarray) -> bytes:
    return struct.pack(f"{len(v)}f", *v.tolist())


def _delete_file_data(db: sqlite3.Connection, file_id: int) -> None:
    """Remove all chunk, vector, and FTS data for a file."""
    db.execute("DELETE FROM document_vectors WHERE file_id=?", (file_id,))
    chunk_rows = db.execute(
        "SELECT id, text FROM chunks WHERE file_id = ?", (file_id,)
    ).fetchall()
    if chunk_rows:
        chunk_ids = [r[0] for r in chunk_rows]
        placeholders = ",".join("?" * len(chunk_ids))
        db.execute(f"DELETE FROM vec_chunks WHERE chunk_id IN ({placeholders})", chunk_ids)
        for cid, text in chunk_rows:
            db.execute(
                "INSERT INTO chunks_fts (chunks_fts, rowid, text) VALUES ('delete', ?, ?)",
                (cid, text),
            )
        db.execute(f"DELETE FROM chunks WHERE id IN ({placeholders})", chunk_ids)


def move_indexed_file(
    source: str | Path, destination: str | Path, db_path: str
) -> bool:
    """Move an indexed file record while preserving chunks and embeddings."""
    source = str(Path(source).expanduser().resolve())
    destination_path = Path(destination).expanduser().resolve()
    destination = str(destination_path)
    db = _connect(db_path)
    try:
        source_row = db.execute(
            "SELECT id FROM files WHERE path=?", (source,)
        ).fetchone()
        if source_row is None:
            return False
        destination_row = db.execute(
            "SELECT id FROM files WHERE path=?", (destination,)
        ).fetchone()
        if destination_row and destination_row[0] != source_row[0]:
            _delete_file_data(db, destination_row[0])
            db.execute("DELETE FROM files WHERE id=?", (destination_row[0],))
        db.execute(
            "UPDATE files SET path=?, filename=?, status='indexed' WHERE id=?",
            (destination, destination_path.name, source_row[0]),
        )
        db.commit()
        print(f"[indexer] Moved index record: {source} → {destination}")
        return True
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def index_file(
    path: str | Path,
    db_path: str,
    model_path: str = "~/ferret/models/bge-small-en",
    *,
    raise_on_error: bool = False,
) -> bool:
    """Full pipeline: extract → chunk → embed → store. Skips if hash unchanged."""
    path = Path(path).resolve()
    if not path.exists():
        print(f"[indexer] File not found, skipping: {path}")
        if raise_on_error:
            raise FileNotFoundError(path)
        return False

    current_hash = hash_file(path)
    if not current_hash:
        if raise_on_error:
            raise OSError(f"could not hash {path}")
        return False

    db = _connect(db_path)
    row = None
    try:
        check_index_model(db, model_path)
        row = db.execute(
            "SELECT id, hash, status FROM files WHERE path = ?", (str(path),)
        ).fetchone()

        if row and row[1] == current_hash and row[2] == "indexed":
            print(f"[indexer] Unchanged, skipping: {path.name}")
            return True

        text = extract(path)
        if not text.strip():
            print(f"[indexer] No text extracted from: {path.name}")
            message = "; ".join(extraction_warnings()) or "No readable text extracted"
            db.execute("""INSERT INTO files(path,hash,filename,status,extraction_warning)
                       VALUES (?,?,?,'error',?) ON CONFLICT(path) DO UPDATE SET
                       status='error', extraction_warning=excluded.extraction_warning""",
                       (str(path), current_hash, path.name, message))
            db.commit()
            if raise_on_error:
                raise ValueError(message)
            return False

        if has_manifest(model_path):
            model = load_model(model_path)
            tokenizer = model.chunk_tokenizer
            # Reserve room for model-specific prefixes and special tokens.
            limit = min(model.spec.max_tokens - 32, 480)
            chunks = chunk_document(text, tokenizer, max_tokens=min(256, limit))
            if len(chunks) > 1:
                # Wider overlapping source windows provide document context
                # without inventing summaries or truncating to the first page.
                contextual = chunk_document(text, tokenizer, max_tokens=limit)
                known = {chunk.content_hash for chunk in chunks}
                chunks.extend(chunk for chunk in contextual if chunk.content_hash not in known)
        else:
            tokenizer, _ = _get_session(model_path)
            chunks = chunk_document(text, tokenizer)
        if not chunks:
            return True

        # Reuse embeddings for byte-identical chunks when an indexed document
        # changes around them. Stale rows were produced by another embedding
        # pipeline and must never be reused.
        reusable: dict[tuple[str, str], deque[int]] = defaultdict(deque)
        if row and row[2] != "stale":
            for chunk_id, old_text, old_hash in db.execute(
                """
                SELECT c.id, c.text, c.content_hash
                FROM chunks c JOIN vec_chunks vc ON vc.chunk_id = c.id
                WHERE c.file_id=?
                ORDER BY c.chunk_index
                """,
                (row[0],),
            ):
                digest = old_hash or chunk_content_hash(old_text)
                reusable[(digest, old_text)].append(chunk_id)

        chunk_rows = []
        texts_to_embed = []
        for chunk_index, chunk in enumerate(chunks):
            digest = chunk.content_hash
            matches = reusable.get((digest, chunk.text))
            reused_id = matches.popleft() if matches else None
            chunk_rows.append((chunk_index, chunk, digest, reused_id))
            if reused_id is None:
                texts_to_embed.append(chunk.text)

        vectors = iter(embed(texts_to_embed, model_path))

        if row:
            file_id = row[0]
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

        db.execute("UPDATE files SET extraction_warning=? WHERE id=?",
                   ("; ".join(extraction_warnings()), file_id))

        reused_ids = {
            reused_id for _, _, _, reused_id in chunk_rows if reused_id is not None
        }
        if row:
            old_ids = {
                old_id for (old_id,) in db.execute(
                    "SELECT id FROM chunks WHERE file_id=?", (file_id,)
                )
            }
            stale_ids = old_ids - reused_ids
            if stale_ids:
                placeholders = ",".join("?" * len(stale_ids))
                for chunk_id, old_text in db.execute(
                    f"SELECT id, text FROM chunks WHERE id IN ({placeholders})",
                    tuple(stale_ids),
                ).fetchall():
                    db.execute(
                        "INSERT INTO chunks_fts (chunks_fts, rowid, text) "
                        "VALUES ('delete', ?, ?)",
                        (chunk_id, old_text),
                    )
                db.execute(
                    f"DELETE FROM vec_chunks WHERE chunk_id IN ({placeholders})",
                    tuple(stale_ids),
                )
                db.execute(
                    f"DELETE FROM chunks WHERE id IN ({placeholders})",
                    tuple(stale_ids),
                )

        for i, chunk, digest, reused_id in chunk_rows:
            if reused_id is not None:
                db.execute(
                    """UPDATE chunks SET chunk_index=?, content_hash=?,
                       page_number=?, start_line=?, end_line=?, heading=? WHERE id=?""",
                    (
                        i, digest, chunk.page_number, chunk.start_line,
                        chunk.end_line, chunk.heading, reused_id,
                    ),
                )
                continue
            vec = next(vectors)
            cursor = db.execute(
                """INSERT INTO chunks
                   (file_id, chunk_index, text, content_hash, page_number,
                    start_line, end_line, heading) VALUES (?,?,?,?,?,?,?,?)""",
                (
                    file_id, i, chunk.text, digest, chunk.page_number,
                    chunk.start_line, chunk.end_line, chunk.heading,
                ),
            )
            chunk_id = cursor.lastrowid
            db.execute(
                "INSERT INTO vec_chunks (chunk_id, embedding) VALUES (?,?)",
                (chunk_id, _serialize_vector(vec)),
            )
            db.execute(
                "INSERT INTO chunks_fts (rowid, text) VALUES (?,?)",
                (chunk_id, chunk.text),
            )
            db.execute("INSERT INTO chunks_ar_fts(rowid,text) VALUES (?,?)",
                       (chunk_id, lexical_text(chunk.text)))

        # A document-level representation provides a separate recall route;
        # long documents no longer compete solely through their best fragment.
        vector_rows = db.execute("""SELECT vc.embedding FROM vec_chunks vc JOIN chunks c ON c.id=vc.chunk_id
                                  WHERE c.file_id=?""", (file_id,)).fetchall()
        if vector_rows:
            pooled = np.mean([np.frombuffer(blob, dtype=np.float32) for (blob,) in vector_rows], axis=0)
            pooled /= max(float(np.linalg.norm(pooled)), 1e-9)
            db.execute("INSERT OR REPLACE INTO document_vectors(file_id,embedding) VALUES (?,?)",
                       (file_id, _serialize_vector(pooled.astype(np.float32))))
        db.commit()
        print(
            f"[indexer] Indexed {path.name}: {len(chunks)} chunks "
            f"({len(reused_ids)} embeddings reused)"
        )
        return True
    except Exception as e:
        print(f"[indexer] Error indexing {path}: {e}")
        db.rollback()
        if row:
            db.execute(
                "UPDATE files SET hash='', status='error' WHERE id=?", (row[0],)
            )
            db.commit()
        if raise_on_error:
            raise
        return False
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


def rebuild_fts(db_path: str) -> None:
    """Rebuild the FTS index from existing chunk data (one-time migration)."""
    db = _connect(db_path)
    # Create the table if it doesn't exist yet
    db.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            text, content='chunks', content_rowid='id'
        )
    """)
    # Clear and rebuild
    db.execute("INSERT INTO chunks_fts (chunks_fts) VALUES ('delete-all')")
    db.execute("INSERT INTO chunks_fts (rowid, text) SELECT id, text FROM chunks")
    db.execute("DELETE FROM chunks_ar_fts")
    db.executemany("INSERT INTO chunks_ar_fts(rowid,text) VALUES (?,?)",
                   ((cid, lexical_text(text)) for cid, text in db.execute("SELECT id,text FROM chunks").fetchall()))
    db.commit()
    count = db.execute("SELECT count(*) FROM chunks").fetchone()[0]
    db.close()
    print(f"[indexer] Rebuilt FTS index for {count} chunks")


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
