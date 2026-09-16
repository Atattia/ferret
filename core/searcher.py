import re
import struct
import sqlite3
from pathlib import Path
from contextvars import ContextVar
from functools import lru_cache
import time
import unicodedata

import numpy as np
import sqlite_vec

from core.indexer import embed, _serialize_vector
from core.query import SearchFilters, escape_like, parse_query
from core.language import normalize, tokens, passage_preview, lexical_text
from core.models import check_index_model, load_model, calibrated_threshold


_RRF_K = 60
# Retrieval maximizes recall. Relevance rejection requires a separately
# calibrated reranker threshold for the chosen model and corpus.
_diagnostics = ContextVar("search_diagnostics", default=None)


def _warning(message):
    target = _diagnostics.get()
    if target is not None:
        target.append(message)
    print(f"[searcher] {message}")
_ROUTE_WEIGHTS = {
    "filename": 2.0,
    "doctype": 2.0,
    "fts": 1.0,
    "semantic": 1.0,
    "normalized": 0.8,
    "document": 1.0,
    "filename_normalized": 1.5,
}


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


def _semantic_query_is_usable(query: str) -> bool:
    """Reject inputs that embeddings will turn into arbitrary neighbors."""
    words = re.findall(r"[^\W\d_]+", query, re.UNICODE)
    if not words:
        return False
    # The old Latin-only heuristic silently rejected every Arabic query.
    if any(any(ord(char) > 127 for char in word) for word in words):
        return True
    meaningful = [token for token in words if len(token) >= 4]
    if not meaningful:
        return True
    vowel_pattern = re.compile(r"[aeiou]", re.IGNORECASE)
    vowel_less = sum(not vowel_pattern.search(token) for token in meaningful)
    repeated = sum(len(set(token.lower())) == 1 for token in meaningful)
    return vowel_less * 2 <= len(meaningful) and repeated == 0


def _vector_distance_metric(db: sqlite3.Connection) -> str:
    """Return the vec table's metric, including pre-cosine Ferret databases."""
    row = db.execute(
        "SELECT sql FROM sqlite_master WHERE name='vec_chunks'"
    ).fetchone()
    schema = (row[0] if row else "").lower().replace(" ", "")
    return "cosine" if "distance_metric=cosine" in schema else "l2"


def _as_cosine_distance(distance: float, metric: str) -> float:
    """Normalize sqlite-vec distance for unit-length Ferret embeddings."""
    distance = float(distance)
    if metric == "cosine":
        return distance
    # For normalized vectors: ||a-b||² = 2(1-cos(a,b)).
    return (distance * distance) / 2.0


def _tokenize_for_filename(query: str) -> list[str]:
    """Break query into individual meaningful words for flexible filename matching."""
    words = query.lower().split()
    return [w for w in words if w not in _FILLER and len(w) > 1]


@lru_cache(maxsize=128)
def _coverage_groups(query):
    # Each original query term gets one vote, even if normalization adds an
    # Arabic article variant. Repetition must not manufacture extra evidence.
    return tuple(frozenset(_fold_fts_word(term) for term in tokens(lexical_text(word)))
                 for word in dict.fromkeys(tokens(query)) if word not in _FILLER)


@lru_cache(maxsize=8192)
def _fold_fts_word(word):
    # SQLite's default unicode61 matcher also folds Latin accents. Do not
    # accidentally suppress those legitimate hits, or erase Arabic hamzas.
    return "".join("".join(part for part in unicodedata.normalize("NFD", char)
                           if unicodedata.category(part) != "Mn")
                   if ord(char) > 127 and "LATIN" in unicodedata.name(char, "") else char
                   for char in word)


def _lexical_coverage(text, groups):
    if not groups:
        return 0.0
    source = {_fold_fts_word(word) for word in tokens(lexical_text(text))}
    return sum(bool(group & source) for group in groups) / len(groups)


def _is_resume_intent(query: str) -> bool:
    """Recognize explicit requests for a résumé/CV document."""
    normalized = query.lower().replace("é", "e")
    tokens = set(re.findall(r"[a-z]+", normalized))
    return bool(
        {"resume", "resumes", "cv", "cvs"} & tokens
        or "curriculum vitae" in normalized
    )


def _connect(db_path: str) -> sqlite3.Connection:
    # Searching must never create or mutate an index, even on a typoed path.
    uri = Path(db_path).expanduser().resolve().as_uri() + "?mode=ro"
    db = sqlite3.connect(uri, uri=True)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    db.execute("PRAGMA case_sensitive_like=ON")
    db.create_function("ferret_normalize", 1, normalize, deterministic=True)
    return db


def _filename_search(db: sqlite3.Connection, query: str, top_k: int,
                     filters: SearchFilters = SearchFilters()) -> list[dict]:
    """
    Flexible filename matching.
    - Exact substring match (highest priority)
    - Individual word matches: "tax return" finds "tax_return_2024.pdf"
    """
    results = []
    seen = set()
    filter_sql, filter_params = filters.sql()

    # 1. Exact substring match
    pattern = f"%{escape_like(query.lower())}%"
    rows = db.execute(
        f"""
        SELECT f.id, f.path, f.filename, c.text
        FROM files f
        LEFT JOIN chunks c ON c.file_id = f.id AND c.chunk_index = 0
        WHERE lower(f.filename) LIKE ? ESCAPE '\\'
          AND f.status = 'indexed'
          {filter_sql}
        ORDER BY length(f.filename), f.filename, f.path
        LIMIT ?
        """,
        (pattern, *filter_params, top_k),
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
        conditions = " AND ".join(["lower(f.filename) LIKE ? ESCAPE '\\'"] * len(words))
        params = [f"%{escape_like(w)}%" for w in words] + filter_params + [top_k]
        rows = db.execute(
            f"""
            SELECT f.id, f.path, f.filename, c.text
            FROM files f
            LEFT JOIN chunks c ON c.file_id = f.id AND c.chunk_index = 0
            WHERE {conditions}
              AND f.status = 'indexed'
              {filter_sql}
            ORDER BY length(f.filename), f.filename, f.path
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
        or_conditions = " OR ".join(["lower(f.filename) LIKE ? ESCAPE '\\'"] * len(words))
        params = [f"%{escape_like(w)}%" for w in words] + filter_params + [top_k * 2]
        rows = db.execute(
            f"""
            SELECT f.id, f.path, f.filename, c.text
            FROM files f
            LEFT JOIN chunks c ON c.file_id = f.id AND c.chunk_index = 0
            WHERE ({or_conditions})
              AND f.status = 'indexed'
              {filter_sql}
            ORDER BY length(f.filename), f.filename, f.path
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
                "query_coverage": match_count / len(words),
            })

    return results[:top_k]


def _fts_search(db: sqlite3.Connection, query: str, top_k: int,
                filters: SearchFilters = SearchFilters()) -> list[dict]:
    """Full-text keyword search over chunk content using SQLite FTS5."""
    # Build an FTS query: each word joined with OR for broad matching
    words = _tokenize_for_filename(query)
    if not words:
        return []

    # FTS5 query: "word1" OR "word2" OR "word3"
    fts_query = " OR ".join('"' + w.replace('"', '""') + '"' for w in words)
    filter_sql, filter_params = filters.sql()

    try:
        rows = db.execute(
            f"""
            SELECT
                c.file_id,
                f.path,
                f.filename,
                snippet(chunks_fts, 0, '»', '«', '…', 40) AS snip,
                bm25(chunks_fts) AS rank,
                c.page_number,
                c.start_line,
                c.end_line,
                c.heading,
                c.text
            FROM chunks_fts
            JOIN chunks c ON c.id = chunks_fts.rowid
            JOIN files f ON f.id = c.file_id
            WHERE chunks_fts MATCH ?
              AND f.status = 'indexed'
              {filter_sql}
            ORDER BY rank
            LIMIT ?
            """,
            (fts_query, *filter_params, top_k * 3),
        ).fetchall()
    except Exception as e:
        print(f"[searcher] FTS search failed: {e}")
        return []

    results = []
    seen_files = set()
    coverage_groups = _coverage_groups(query)
    for file_id, path, filename, snippet, rank, page, start_line, end_line, heading, source_text in rows:
        if file_id in seen_files:
            continue
        if not Path(path).exists():
            continue
        seen_files.add(file_id)
        snippet = (snippet or "")[:300].replace("\n", " ").strip()
        # BM25 scores are negative (lower = better), normalize to 0-1 range
        score = round(min(0.85, max(0.5, 1.0 + rank * 0.05)), 4)
        results.append({
            "filename": filename,
            "path": path,
            "snippet": snippet,
            "score": score,
            "page": page,
            "start_line": start_line,
            "end_line": end_line,
            "heading": heading,
            "query_coverage": _lexical_coverage(source_text, coverage_groups),
        })
        if len(results) >= top_k:
            break

    return results


def _document_type_search(
    db: sqlite3.Connection, query: str, top_k: int,
    filters: SearchFilters = SearchFilters(),
) -> list[dict]:
    """Retrieve structurally recognizable document types for explicit intents."""
    if not _is_resume_intent(query):
        return []

    filter_sql, filter_params = filters.sql()
    rows = db.execute(
        f"""
        WITH resume_signals AS (
            SELECT
                f.id AS file_id,
                f.path,
                f.filename,
                MAX(CASE WHEN c.chunk_index = 0 THEN c.text END) AS snippet,
                MAX(instr(lower(c.text), 'experience') > 0) AS experience,
                MAX(instr(lower(c.text), 'employment') > 0) AS employment,
                MAX(instr(lower(c.text), 'education') > 0) AS education,
                MAX(instr(lower(c.text), 'skills') > 0) AS skills,
                MAX(instr(lower(c.text), 'projects') > 0) AS projects,
                MAX(instr(lower(c.text), 'certifications') > 0) AS certifications,
                MAX(instr(c.text, '@') > 0) AS email
            FROM files f
            JOIN chunks c ON c.file_id = f.id
            WHERE f.status = 'indexed'
              {filter_sql}
            GROUP BY f.id
        )
        SELECT path, filename, snippet,
               (experience * 3 + employment * 2 + education * 3 +
                skills * 2 + projects + certifications + email) AS type_score
        FROM resume_signals
        WHERE (experience = 1 OR employment = 1)
          AND (education = 1 OR skills = 1 OR projects = 1 OR certifications = 1)
        ORDER BY type_score DESC, filename, path
        LIMIT ?
        """,
        (*filter_params, top_k),
    ).fetchall()

    results = []
    for path, filename, snippet, type_score in rows:
        if not Path(path).exists():
            continue
        results.append({
            "filename": filename,
            "path": path,
            "snippet": (snippet or "")[:300].replace("\n", " ").strip(),
            "score": float(type_score),
        })
    return results


def _normalized_search(db, query, top_k, filters):
    if not any("\u0600" <= char <= "\u06ff" or char.isdigit() for char in query):
        return []  # Do not count identical English BM25 evidence twice.
    words = list(dict.fromkeys(tokens(lexical_text(query))))[:32]
    stopwords = {"في", "من", "على", "عن", "الي", "الى", "مع", "هذا", "هذه", "الذي", "التي", "هو", "هي"}
    words = [word for word in words if word not in stopwords and word not in _FILLER]
    if not words:
        return []
    expression = " OR ".join('"' + word + '"' for word in words)
    clause, params = filters.sql()
    try:
        rows = db.execute(f"""
            SELECT f.path,f.filename,c.text,c.page_number,c.start_line,c.end_line,c.heading
            FROM chunks_ar_fts JOIN chunks c ON c.id=chunks_ar_fts.rowid
            JOIN files f ON f.id=c.file_id
            WHERE chunks_ar_fts MATCH ? AND f.status='indexed' {clause}
            ORDER BY bm25(chunks_ar_fts) LIMIT ?
        """, (expression, *params, top_k * 5)).fetchall()
    except sqlite3.OperationalError as exc:
        if "no such table" in str(exc):
            _warning("Arabic keyword index unavailable; restart Ferret to upgrade the index.")
            return []
        raise
    results, seen = [], set()
    coverage_groups = _coverage_groups(query)
    for path, filename, text, page, start, end, heading in rows:
        if path in seen or not Path(path).is_file():
            continue
        seen.add(path)
        results.append(dict(path=path, filename=filename, snippet=passage_preview(text, query),
                            query_coverage=_lexical_coverage(text, coverage_groups),
                            score=1.0, page=page, start_line=start, end_line=end, heading=heading,
                            passages=[dict(text=text, page=page, start_line=start, end_line=end, heading=heading)]))
        if len(results) >= top_k:
            break
    return results


def _normalized_filename_search(db, query, top_k, filters):
    if not any("\u0600" <= char <= "\u06ff" or char.isdigit() for char in query):
        return []
    words = list(dict.fromkeys(tokens(query)))[:16]
    if not words:
        return []
    clause, params = filters.sql()
    conditions = " AND ".join("ferret_normalize(f.filename) LIKE ? ESCAPE '\\'" for _ in words)
    rows = db.execute(f"""SELECT f.path,f.filename,c.text FROM files f
                       LEFT JOIN chunks c ON c.file_id=f.id AND c.chunk_index=0
                       WHERE f.status='indexed' AND {conditions} {clause}
                       ORDER BY f.filename,f.path LIMIT ?""",
                      (*["%" + escape_like(word) + "%" for word in words], *params, top_k)).fetchall()
    return [dict(path=path, filename=name, snippet=(text or "")[:300], score=.9)
            for path, name, text in rows if Path(path).is_file()]


def _semantic_search(
    db: sqlite3.Connection,
    query: str,
    top_k: int,
    model_path: str,
    filters: SearchFilters = SearchFilters(),
) -> list[dict]:
    """Return file-deduplicated semantic candidates for one query."""
    # Preserve negation and relationships for the language model. Lexical
    # retrieval may remove filler words; semantic retrieval needs the sentence.
    cleaned = query.strip()
    if not _semantic_query_is_usable(cleaned):
        print("[searcher] Semantic query rejected as non-linguistic")
        return []

    try:
        check_index_model(db, model_path)
        # Queries and passages have different encoding conventions for BGE.
        query_vec = embed([cleaned], model_path, is_query=True)[0]
        query_bytes = _serialize_vector(query_vec)
    except Exception as e:
        _warning(f"Semantic search unavailable: {e}")
        return []

    filter_sql, filter_params = filters.sql()
    filtered = bool(filter_sql)
    # Filter before top-k: globally nearest chunks can all be outside the
    # requested folder/type. Exact cosine search over eligible rows guarantees
    # recall for scoped queries, including databases using the legacy L2 metric.
    distance_sql = "vec_distance_cosine(vc.embedding, ?)" if filtered else "vc.distance"
    where_sql = "" if filtered else "vc.embedding MATCH ? AND k = ? AND"
    params = ((query_bytes, *filter_params, max(top_k * 5, top_k)) if filtered
              else (query_bytes, max(top_k * 5, top_k)))
    try:
        rows = db.execute(
            f"""
            SELECT
                vc.chunk_id,
                {distance_sql} AS distance,
                c.text,
                f.path,
                f.filename,
                f.hash,
                f.id AS file_id,
                c.page_number,
                c.start_line,
                c.end_line,
                c.heading
            FROM vec_chunks vc
            JOIN chunks c ON c.id = vc.chunk_id
            JOIN files f ON f.id = c.file_id
            WHERE {where_sql} f.status = 'indexed'
              {filter_sql}
            ORDER BY distance
            {"LIMIT ?" if filtered else ""}
            """,
            params,
        ).fetchall()
    except Exception as e:
        _warning(f"Vector search failed: {e}")
        return []

    results = []
    seen_paths = set()
    by_path = {}
    metric = "cosine" if filtered else _vector_distance_metric(db)
    for (
        chunk_id, distance, text, path, filename, file_hash, file_id,
        page, start_line, end_line, heading,
    ) in rows:
        cosine_distance = _as_cosine_distance(distance, metric)
        if path in seen_paths:
            if len(by_path[path]["passages"]) < 3:
                by_path[path]["passages"].append(dict(text=text, page=page, start_line=start_line,
                                                      end_line=end_line, heading=heading))
            continue

        path_obj = Path(path)
        if not path_obj.exists():
            # The watcher/reconciler owns file lifecycle. Never recursively
            # scan the user's home or compete with index writes during search.
            continue

        seen_paths.add(path)
        results.append({
            "filename": filename,
            "path": path,
            "snippet": text[:300].replace("\n", " ").strip(),
            # Preserve the backend score for diagnostics. RRF uses rank, not
            # this value, because distances and BM25 scores are incomparable.
            "score": 1.0 - cosine_distance,
            "page": page,
            "start_line": start_line,
            "end_line": end_line,
            "heading": heading,
            "semantic_distance": cosine_distance,
            "passages": [dict(text=text, page=page, start_line=start_line, end_line=end_line, heading=heading)],
        })
        by_path[path] = results[-1]

    return results[:top_k]


def _document_semantic_search(db, query, top_k, model_path, filters):
    if not _semantic_query_is_usable(query):
        return []
    try:
        check_index_model(db, model_path)
        vector = _serialize_vector(embed([query], model_path, is_query=True)[0])
        clause, params = filters.sql()
        rows = db.execute(f"""SELECT f.path,f.filename,c.text,vec_distance_cosine(d.embedding,?) AS distance
                            FROM document_vectors d JOIN files f ON f.id=d.file_id
                            LEFT JOIN chunks c ON c.file_id=f.id AND c.chunk_index=0
                            WHERE f.status='indexed' {clause} ORDER BY distance LIMIT ?""",
                          (vector, *params, top_k)).fetchall()
        return [dict(path=path, filename=filename, snippet=(text or "")[:300], score=1-distance)
                for path, filename, text, distance in rows if Path(path).is_file()]
    except Exception:
        # Legacy indexes have no document vectors. Passage retrieval already
        # supplies model diagnostics; avoid reporting the same failure twice.
        return []


def _fuse_ranked_results(
    routes: list[tuple[str, list[dict]]],
    top_k: int,
) -> list[dict]:
    """Fuse independent result lists with weighted reciprocal rank fusion."""
    candidates: dict[str, dict] = {}

    for route_name, route_results in routes:
        weight = _ROUTE_WEIGHTS[route_name]
        for rank, result in enumerate(route_results, start=1):
            path = result["path"]
            candidate = candidates.get(path)
            if candidate is None:
                candidate = {
                    "filename": result["filename"],
                    "path": path,
                    "snippet": result.get("snippet", ""),
                    "score": 0.0,
                    "matched_by": [],
                    "evidence": [],
                }
                for field in ("page", "start_line", "end_line", "heading"):
                    if result.get(field) is not None:
                        candidate[field] = result[field]
                candidates[path] = candidate

            # OR-based retrieval is deliberately broad. A single incidental
            # term must not cast the same vote as matching the whole query.
            coverage = result.get("query_coverage", 1.0)
            candidate["score"] += weight * coverage / (_RRF_K + rank)
            if result.get("passages"):
                passages = candidate.setdefault("passages", [])
                for passage in result["passages"]:
                    if not any(p["text"] == passage["text"] for p in passages):
                        passages.append(passage)
                candidate["passages"] = passages[:4]
            if "semantic_distance" in result:
                candidate["semantic_distance"] = result["semantic_distance"]
            candidate["matched_by"].append(route_name)
            snippet = result.get("snippet", "")
            if snippet and not any(
                evidence["snippet"] == snippet for evidence in candidate["evidence"]
            ):
                candidate["evidence"].append({
                    "route": route_name,
                    "snippet": snippet,
                    "rank": rank,
                    "query_coverage": coverage,
                })

            # A content hit is normally a more useful preview than the first
            # chunk selected by filename matching.
            if route_name in {"doctype", "fts", "semantic"} and snippet:
                candidate["snippet"] = snippet
                for field in ("page", "start_line", "end_line", "heading"):
                    if result.get(field) is not None:
                        candidate[field] = result[field]

    ranked = sorted(candidates.values(), key=lambda item: (-item["score"], item["path"]))
    if not ranked:
        return []

    # Expose a stable relative 0..1 score to the UI while retaining rank-based
    # fusion internally. Raw scores from different retrieval systems must not
    # be compared directly.
    best_score = ranked[0]["score"]
    for candidate in ranked:
        candidate["score"] = round(candidate["score"] / best_score, 4) if best_score else 0.0
        candidate["evidence"] = candidate["evidence"][:3]

    return ranked[:top_k]


def _rerank(query, candidates, model_path, budget=40, minimum=None, cancelled=None):
    model = load_model(model_path)
    pairs = []
    # Round-robin gives each document a chance before its additional passages.
    for position in range(3):
        for candidate in candidates:
            passages = candidate.get("passages") or [{"text": candidate["snippet"]}]
            if position < len(passages):
                pairs.append((candidate, passages[position]))
            if len(pairs) >= budget:
                break
        if len(pairs) >= budget:
            break
    kwargs = {"cancelled": cancelled} if cancelled else {}
    scores = model.rerank(query, [passage["text"] for _, passage in pairs], **kwargs)
    chosen = {}
    for (candidate, passage), score in zip(pairs, scores):
        path = candidate["path"]
        if path not in chosen or score > chosen[path]["rerank_score"]:
            result = dict(candidate, rerank_score=score)
            result["snippet"] = passage_preview(passage["text"], query)
            for key in ("page", "start_line", "end_line", "heading"):
                result.pop(key, None)
                if passage.get(key) is not None:
                    result[key] = passage[key]
            chosen[path] = result
    ranked = sorted(chosen.values(), key=lambda r: (-r["rerank_score"], -r["score"], r["path"]))
    retrieval_ranks = {candidate["path"]: rank for rank, candidate in enumerate(candidates, 1)}
    for rank, result in enumerate(ranked, 1):
        # Rank-based blending lets a weaker cross-encoder refine a stronger
        # retriever without erasing all of its evidence. Logits remain separate
        # for calibration; neither system's raw score is treated as comparable.
        result["score"] = 2 / (_RRF_K + rank) + 1 / (_RRF_K + retrieval_ranks[result["path"]])
        result["matched_by"] = [*result["matched_by"], "reranker"]
    ranked.sort(key=lambda r: (-r["score"], r["path"]))
    if ranked:
        best = ranked[0]["score"]
        for result in ranked:
            result["score"] = round(result["score"] / best, 4)
    # Raw logits are deliberately not presented as probabilities. The optional
    # threshold must be chosen on development data for this exact model.
    return [r for r in ranked if minimum is None or r["rerank_score"] >= minimum]


def _deduplicate_copies(db, candidates):
    """Identical files should not occupy several result/reranking slots."""
    if not candidates:
        return []
    paths = [r["path"] for r in candidates]
    placeholders = ",".join("?" for _ in paths)
    hashes = dict(db.execute(f"SELECT path,hash FROM files WHERE path IN ({placeholders})", paths).fetchall())
    by_hash, unique = {}, []
    for candidate in candidates:
        digest = hashes.get(candidate["path"])
        if digest and digest in by_hash:
            first = by_hash[digest]
            first.setdefault("copies", [first["path"]]).append(candidate["path"])
            continue
        unique.append(candidate)
        if digest:
            by_hash[digest] = candidate
    return unique


def search(
    query: str,
    db_path: str,
    top_k: int = 5,
    model_path: str = "~/ferret/models/bge-small-en",
    *,
    mode: str = "hybrid",
    reranker_path: str | None = None,
    diagnostics: list | None = None,
    rerank_minimum: float | None = None,
    rerank_budget: int = 40,
    calibration_path: str | None = None,
    on_candidates=None,
    cancelled=None,
    on_ranked=None,
) -> list[dict]:
    """
    Hybrid search combining filename, document-type, keyword, and semantic recall.

    Candidate lists are fused using weighted reciprocal rank fusion (RRF), so
    strong results from one route are not hidden merely because another route
    filled ``top_k`` first.
    """
    if mode not in {"hybrid", "keyword", "semantic"}:
        raise ValueError("mode must be hybrid, keyword, or semantic")
    if top_k <= 0 or not query.strip():
        return []
    top_k = min(top_k, 100)
    query, filters = parse_query(query)
    token = _diagnostics.set(diagnostics)

    try:
        db = _connect(db_path)
    except Exception as e:
        _warning(f"Index unavailable: {e}")
        _diagnostics.reset(token)
        return []

    candidate_depth = max(top_k * 5, 100)
    try:
        if cancelled and cancelled():
            return []
        if not query:
            return _fuse_ranked_results([
                ("filename", _filename_search(db, "", candidate_depth, filters))
            ], top_k)
        filename_results = (_filename_search(db, query, candidate_depth, filters)
                            if mode != "semantic" else [])
        document_type_results = (_document_type_search(db, query, candidate_depth, filters)
                                 if mode != "semantic" else [])
        fts_results = (_fts_search(db, query, candidate_depth, filters)
                       if mode != "semantic" else [])
        normalized_results = (_normalized_search(db, query, candidate_depth, filters)
                              if mode != "semantic" else [])
        normalized_names = (_normalized_filename_search(db, query, candidate_depth, filters)
                            if mode != "semantic" else [])
        if cancelled and cancelled():
            return []
        semantic_results = (_semantic_search(db, query, candidate_depth, model_path, filters)
                            if mode != "keyword" else [])
        if cancelled and cancelled():
            return []
        document_results = (_document_semantic_search(db, query, candidate_depth, model_path, filters)
                            if mode != "keyword" else [])
        candidates = _fuse_ranked_results(
            [
                ("filename", filename_results),
                ("doctype", document_type_results),
                ("fts", fts_results),
                ("semantic", semantic_results),
                ("normalized", normalized_results),
                ("document", document_results),
                ("filename_normalized", normalized_names),
            ],
            candidate_depth,
        )
        candidates = _deduplicate_copies(db, candidates)
        if cancelled and cancelled():
            return []
        use_reranker = bool(mode != "keyword" and reranker_path and len(tokens(query)) >= 3)
        if on_candidates and use_reranker:
            on_candidates([{key: value for key, value in result.items() if key != "passages"}
                           for result in candidates[:top_k]])
        if use_reranker and candidates:
            try:
                if calibration_path:
                    rerank_minimum = calibrated_threshold(calibration_path, model_path, reranker_path)
                candidates = _rerank(query, candidates, reranker_path, max(top_k, rerank_budget), rerank_minimum,
                                     cancelled=cancelled)
            except Exception as exc:
                _warning(f"Reranking unavailable; showing hybrid ranking: {exc}")
        if cancelled and cancelled():
            return []
        for candidate in candidates:
            candidate.pop("passages", None)
        if on_ranked:
            on_ranked(candidates)
        return candidates[:top_k]
    finally:
        db.close()
        _diagnostics.reset(token)


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
