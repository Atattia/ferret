import struct
import sqlite3
from pathlib import Path

import numpy as np
import sqlite_vec

from core.indexer import embed, _serialize_vector


def _connect(db_path: str) -> sqlite3.Connection:
    db = sqlite3.connect(str(Path(db_path).expanduser()))
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    return db


def _filename_search(db: sqlite3.Connection, query: str, top_k: int) -> list[dict]:
    """Return indexed files whose filename contains the query (case-insensitive)."""
    pattern = f"%{query.lower()}%"
    rows = db.execute(
        """
        SELECT f.id, f.path, f.filename, c.text
        FROM files f
        LEFT JOIN chunks c ON c.file_id = f.id AND c.chunk_index = 0
        WHERE lower(f.filename) LIKE ?
          AND f.status = 'indexed'
        LIMIT ?
        """,
        (pattern, top_k),
    ).fetchall()

    results = []
    for file_id, path, filename, text in rows:
        if not Path(path).exists():
            continue
        snippet = (text or "")[:300].replace("\n", " ").strip()
        results.append({
            "filename": filename,
            "path": path,
            "snippet": snippet,
            "score": 1.0,
        })
    return results


def search(
    query: str,
    db_path: str,
    top_k: int = 5,
    model_path: str = "~/ferret/models/bge-small-en",
) -> list[dict]:
    """
    Filename match (priority) + semantic vector search → top_k results.
    Each dict has: filename, path, snippet, score.
    Filename matches always rank first with score=1.0.
    """
    if not query.strip():
        return []

    try:
        db = _connect(db_path)
    except Exception as e:
        print(f"[searcher] Failed to connect to DB: {e}")
        return []

    # --- Filename matches (highest priority) ---
    results = _filename_search(db, query, top_k)
    seen_paths = {r["path"] for r in results}

    if len(results) >= top_k:
        db.close()
        return results[:top_k]

    # --- Semantic search to fill remaining slots ---
    semantic_slots = top_k - len(results)

    try:
        query_vec = embed([query], model_path)[0]
        query_bytes = _serialize_vector(query_vec)
    except Exception as e:
        print(f"[searcher] Failed to embed query: {e}")
        db.close()
        return results

    try:
        rows = db.execute(
            """
            SELECT
                vc.chunk_id,
                vc.distance,
                c.text,
                f.path,
                f.filename,
                f.hash,
                f.id AS file_id
            FROM vec_chunks vc
            JOIN chunks c ON c.id = vc.chunk_id
            JOIN files f ON f.id = c.file_id
            WHERE vc.embedding MATCH ?
              AND k = ?
            ORDER BY vc.distance
            """,
            (query_bytes, semantic_slots * 3),
        ).fetchall()
    except Exception as e:
        print(f"[searcher] Vector search failed: {e}")
        db.close()
        return results

    for chunk_id, distance, text, path, filename, file_hash, file_id in rows:
        if path in seen_paths:
            continue

        path_obj = Path(path)
        if not path_obj.exists():
            relocated = _find_by_hash(db, file_hash, path)
            if relocated:
                db.execute(
                    "UPDATE files SET path=?, filename=? WHERE id=?",
                    (str(relocated), relocated.name, file_id),
                )
                db.commit()
                path = str(relocated)
                filename = relocated.name
                print(f"[searcher] Relocated file: {filename} → {path}")
            else:
                db.execute(
                    "UPDATE files SET status='orphaned' WHERE id=?", (file_id,)
                )
                db.commit()
                print(f"[searcher] Marked orphaned: {filename}")
                continue

        seen_paths.add(path)
        snippet = text[:300].replace("\n", " ").strip()
        score = 1.0 - float(distance)

        results.append({
            "filename": filename,
            "path": path,
            "snippet": snippet,
            "score": round(score, 4),
        })

        if len(results) >= top_k:
            break

    db.close()
    return results


def _find_by_hash(db: sqlite3.Connection, file_hash: str, original_path: str) -> Path | None:
    """Attempt to find a file by its hash by searching the parent directory tree."""
    if not file_hash:
        return None

    from core.hasher import hash_file

    original = Path(original_path)
    filename = original.name

    # Search common locations: same parent, home dir
    search_roots = [
        original.parent,
        Path.home(),
        Path.home() / "Documents",
        Path.home() / "Downloads",
        Path.home() / "Desktop",
    ]

    for root in search_roots:
        if not root.exists():
            continue
        try:
            for candidate in root.rglob(filename):
                if hash_file(candidate) == file_hash:
                    return candidate
        except Exception:
            continue

    return None
