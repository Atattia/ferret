"""Read-only re-embedding experiment against the user's existing corpus.

No document text or unrelated filenames are included in the report. The existing
index is never modified. Only the supplied test filename is evaluated.
"""
import argparse
import json
from pathlib import Path
import time
import hashlib
from collections import defaultdict

import numpy as np

from core.indexer import embed
from core.models import load_model, model_spec
from core.searcher import _connect
from core.maintenance import atomic_json

QUERIES = ["Employee burnout and staff retention problems",
           "Why workplace perks cannot compensate for excessive workloads",
           "burnout", "retention", "workloads",
           "موظفون يتركون العمل بسبب الإرهاق والضغط المستمر",
           "الناس بتمشي من الشغل عشان مش قادرة تستحمل الضغط"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/settings.json")
    parser.add_argument("--model", required=True)
    parser.add_argument("--reranker")
    parser.add_argument("--filename", default="note-074.txt")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Match passage-sized production inference rather than full model context")
    parser.add_argument("--sample-per-file", type=int,
                        help="Explicitly sample evenly spaced passages per document for a cheaper diagnostic")
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    from core.models import has_manifest
    if has_manifest(args.model):
        load_model(args.model).tokenizer.enable_truncation(max_length=args.max_tokens)
    config = json.loads(Path(args.config).read_text())
    db = _connect(config["db_path"])
    try:
        records = db.execute("""SELECT f.path,f.filename,c.text FROM chunks c
                               JOIN files f ON f.id=c.file_id WHERE f.status='indexed'
                               ORDER BY c.id""").fetchall()
    finally:
        db.close()
    records = [row for row in records if Path(row[0]).is_file()]
    source_chunks = len(records)
    if args.sample_per_file:
        grouped = defaultdict(list)
        for row in records:
            grouped[row[0]].append(row)
        records = [rows[i] for rows in grouped.values()
                   for i in sorted(set(np.linspace(0, len(rows)-1, min(args.sample_per_file, len(rows)), dtype=int)))]
    if not any(row[1] == args.filename for row in records):
        raise ValueError("Test file is not indexed")
    started = time.perf_counter()
    digest = hashlib.sha256((model_spec(args.model).fingerprint + str(args.max_tokens) + repr(records)).encode()).hexdigest()
    cache = Path("data/probe-cache") / (digest + ".npy")
    if cache.exists():
        vectors = np.load(cache, allow_pickle=False)
    else:
        blocks = []
        for start in range(0, len(records), 32):
            blocks.append(embed([row[2] for row in records[start:start + 32]], args.model))
            print(f"Embedded {min(start + 32, len(records))}/{len(records)} passages", flush=True)
        vectors = np.concatenate(blocks)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, vectors, allow_pickle=False)
    file_indices = defaultdict(list)
    for index, (path, _, _) in enumerate(records):
        file_indices[path].append(index)
    paths = list(file_indices)
    centroids = np.array([vectors[file_indices[path]].mean(0) for path in paths])
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True).clip(min=1e-9)
    embedding_seconds = time.perf_counter() - started
    results = []
    reranker = load_model(args.reranker) if args.reranker else None
    for query in QUERIES:
        vector = embed([query], args.model, is_query=True)[0]
        ordering = np.argsort(-(vectors @ vector))
        seen, candidates = set(), []
        for index in ordering:
            path, filename, text = records[index]
            if path in seen:
                continue
            seen.add(path)
            candidates.append((path, filename, text))
        rank = next((i for i, (_, name, _) in enumerate(candidates, 1) if name == args.filename), None)
        doc_order = [paths[i] for i in np.argsort(-(centroids @ vector))]
        doc_ranks = {path: i for i, path in enumerate(doc_order, 1)}
        passage_ranks = {path: i for i, (path, _, _) in enumerate(candidates, 1)}
        document_rank = min(doc_ranks[path] for path, name, _ in candidates if name == args.filename)
        candidates.sort(key=lambda item: -(1 / (60 + doc_ranks[item[0]]) + 1 / (60 + passage_ranks[item[0]])))
        fused_rank = next((i for i, (_, name, _) in enumerate(candidates, 1) if name == args.filename), None)
        rerank = None
        if reranker:
            selection = candidates[:40]
            from core.language import tokens
            if len(tokens(query)) < 3:
                order = list(range(len(selection)))
            else:
                scores = reranker.rerank(query, [text for _, _, text in selection])
                order = sorted(range(len(scores)), key=lambda i: -scores[i])
                cross_ranks = {position: rank for rank, position in enumerate(order, 1)}
                order.sort(key=lambda position: -(2 / (60 + cross_ranks[position]) + 1 / (61 + position)))
            rerank = next((i for i, position in enumerate(order, 1)
                           if selection[position][1] == args.filename), None)
        results.append(dict(query=query, semantic_rank=rank, document_rank=document_rank,
                            fused_rank=fused_rank, reranked_rank=rerank))
        print(results[-1], flush=True)
    atomic_json(args.report, dict(model=model_spec(args.model).__dict__, chunks=len(records),
                                 embedding_fingerprint=model_spec(args.model).fingerprint,
                                 source_chunks=source_chunks, sample_per_file=args.sample_per_file,
                                 max_tokens=args.max_tokens, embedding_seconds=embedding_seconds, results=results))


if __name__ == "__main__":
    main()
