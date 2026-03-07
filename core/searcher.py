import re
import struct
import sqlite3
from pathlib import Path

import numpy as np
import sqlite_vec

from core.indexer import embed, _serialize_vector


# Words that dilute embedding quality, no support for time searching yet
_FILLER = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "must",
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "it",
    "they", "them", "their", "its", "his", "her",
    "this", "that", "these", "those", "what", "which", "who", "whom",
    "some", "any", "no", "not", "all", "each", "every", "both",
    "about", "from", "into", "with", "without", "for", "of", "on", "in",
    "at", "to", "by", "up", "down", "out", "off", "over", "under",
    "just", "also", "very", "really", "quite", "too", "so", "then",
    "here", "there", "where", "when", "how", "why",
    "find", "search", "look", "looking", "show", "give", "get",
    "file", "files", "document", "documents", "doc", "docs",
    "thing", "stuff", "something",
    "last", "recent", "recently", "ago", "yesterday", "today", "week",
    "month", "year", "downloaded", "saved", "wrote", "created", "made",
}


def _clean_query(query: str) -> str:
    """Strip filler words to sharpen the embedding signal."""
    words = query.lower().split()
    cleaned = [w for w in words if w not in _FILLER and len(w) > 1]
    # If cleaning removes everything, fall back to original
    return " ".join(cleaned) if cleaned else query


def _tokenize_for_filename(query: str) -> list[str]:
    """Break query into individual meaningful words for flexible filename matching."""
    words = query.lower().split()
    return [w for w in words if w not in _FILLER and len(w) > 1]


def _connect(db_path: str) -> sqlite3.Connection:
    db = sqlite3.connect(str(Path(db_path).expanduser()))
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    return db


def _filename_search(db: sqlite3.Connection, query: str, top_k: int) -> list[dict]:
    """
    Flexible filename matching.
    - Exact substring match (highest priority)
    - Individual word matches: "tax return" finds "tax_return_2024.pdf"
    """
    results = []
    seen = set()

    # 1. Exact substring match
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

    for file_id, path, filename, text in rows:
        if not Path(path).exists() or path in seen:
            continue
        seen.add(path)
        snippet = (text or "")[:300].replace("\n", " ").strip()
        results.append({
            "filename": filename,
            "path": path,
            "snippet": snippet,
            "score": 1.0,
        })

    if len(results) >= top_k:
        return results[:top_k]

    # 2. Per-word fuzzy match: each query word must appear somewhere in filename
    words = _tokenize_for_filename(query)
    if len(words) >= 2:
        # Build WHERE clause: lower(f.filename) LIKE '%word1%' AND ... LIKE '%word2%'
        conditions = " AND ".join(["lower(f.filename) LIKE ?"] * len(words))
        params = [f"%{w}%" for w in words] + [top_k]
        rows = db.execute(
            f"""
            SELECT f.id, f.path, f.filename, c.text
            FROM files f
            LEFT JOIN chunks c ON c.file_id = f.id AND c.chunk_index = 0
            WHERE {conditions}
              AND f.status = 'indexed'
            LIMIT ?
            """,
            params,
        ).fetchall()

        for file_id, path, filename, text in rows:
            if not Path(path).exists() or path in seen:
                continue
            seen.add(path)
            snippet = (text or "")[:300].replace("\n", " ").strip()
            results.append({
                "filename": filename,
                "path": path,
                "snippet": snippet,
                "score": 0.95,
            })

    if len(results) >= top_k:
        return results[:top_k]

    # 3. Any single word match (weaker signal)
    if words:
        or_conditions = " OR ".join(["lower(f.filename) LIKE ?"] * len(words))
        params = [f"%{w}%" for w in words] + [top_k * 2]
        rows = db.execute(
            f"""
            SELECT f.id, f.path, f.filename, c.text
            FROM files f
            LEFT JOIN chunks c ON c.file_id = f.id AND c.chunk_index = 0
            WHERE ({or_conditions})
              AND f.status = 'indexed'
            LIMIT ?
            """,
            params,
        ).fetchall()

        for file_id, path, filename, text in rows:
            if not Path(path).exists() or path in seen:
                continue
            seen.add(path)
            # Score by how many query words match the filename
            fname_lower = filename.lower()
            match_count = sum(1 for w in words if w in fname_lower)
            snippet = (text or "")[:300].replace("\n", " ").strip()
            results.append({
                "filename": filename,
                "path": path,
                "snippet": snippet,
                "score": round(0.8 + 0.1 * (match_count / len(words)), 4),
            })

    return results[:top_k]


def _fts_search(db: sqlite3.Connection, query: str, top_k: int, seen_paths: set) -> list[dict]:
    """Full-text keyword search over chunk content using SQLite FTS5."""
    # Build an FTS query: each word joined with OR for broad matching
    words = _tokenize_for_filename(query)
    if not words:
        return []

    # FTS5 query: "word1" OR "word2" OR "word3"
    fts_query = " OR ".join(f'"{w}"' for w in words)

    try:
        rows = db.execute(
            """
            SELECT
                c.file_id,
                f.path,
                f.filename,
                snippet(chunks_fts, 0, '»', '«', '…', 40) AS snip,
                bm25(chunks_fts) AS rank
            FROM chunks_fts
            JOIN chunks c ON c.id = chunks_fts.rowid
            JOIN files f ON f.id = c.file_id
            WHERE chunks_fts MATCH ?
              AND f.status = 'indexed'
            ORDER BY rank
            LIMIT ?
            """,
            (fts_query, top_k * 3),
        ).fetchall()
    except Exception as e:
        print(f"[searcher] FTS search failed: {e}")
        return []

    results = []
    seen_files = set()
    for file_id, path, filename, snippet, rank in rows:
        if path in seen_paths or file_id in seen_files:
            continue
        if not Path(path).exists():
            continue
        seen_files.add(file_id)
        seen_paths.add(path)
        snippet = (snippet or "")[:300].replace("\n", " ").strip()
        # BM25 scores are negative (lower = better), normalize to 0-1 range
        score = round(min(0.85, max(0.5, 1.0 + rank * 0.05)), 4)
        results.append({
            "filename": filename,
            "path": path,
            "snippet": snippet,
            "score": score,
        })
        if len(results) >= top_k:
            break

    return results


def search(
    query: str,
    db_path: str,
    top_k: int = 5,
    model_path: str = "~/ferret/models/bge-small-en",
) -> list[dict]:
    """
    Multi-signal search combining filename, keyword, and semantic matching.

    Priority order:
    1. Filename matches (exact substring, then per-word fuzzy)
    2. Full-text keyword matches in document content (BM25)
    3. Semantic vector search (cosine similarity)

    Results are merged and deduplicated by file path.
    """
    if not query.strip():
        return []

    try:
        db = _connect(db_path)
    except Exception as e:
        print(f"[searcher] Failed to connect to DB: {e}")
        return []

    # --- 1. Filename matches (highest priority) ---
    results = _filename_search(db, query, top_k)
    seen_paths = {r["path"] for r in results}

    if len(results) >= top_k:
        db.close()
        return results[:top_k]

    # --- 2. Full-text keyword search ---
    remaining = top_k - len(results)
    fts_results = _fts_search(db, query, remaining, seen_paths)
    results.extend(fts_results)

    if len(results) >= top_k:
        db.close()
        return results[:top_k]

    # --- 3. Semantic search to fill remaining slots ---
    remaining = top_k - len(results)
    cleaned = _clean_query(query)
    print(f"[searcher] Semantic query: '{query}' → '{cleaned}'")

    try:
        query_vec = embed([cleaned], model_path)[0]
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
            (query_bytes, remaining * 5),
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
