"""Run real-model bilingual relevance checks: python -m benchmarks.run --help."""
import argparse
from collections import defaultdict
from contextlib import redirect_stdout
import json
import math
from pathlib import Path
import statistics
import sys
import tempfile
import time

from benchmarks.corpus import materialize
from core.indexer import init_db, index_file
from core.models import model_spec, RANKING_VERSION
from core.searcher import search


def summarize(rows):
    answered = [row for row in rows if row["relevant"]]
    negatives = [row for row in rows if not row["relevant"]]
    mean = lambda values: statistics.mean(values) if values else None
    timings = sorted(row["elapsed_ms"] for row in rows)
    return {
        "queries": len(rows), "answerable": len(answered),
        "candidate_hit_at_100": mean([row["candidate_hit"] for row in answered]),
        "hit_at_5": mean([row["rank"] is not None and row["rank"] <= 5 for row in answered]),
        "mrr_at_10": mean([1 / row["rank"] if row["rank"] and row["rank"] <= 10 else 0 for row in answered]),
        "no_answer_false_positive_rate": mean([bool(row["results"]) for row in negatives]),
        "p50_ms": statistics.median(timings) if timings else None,
        "p95_ms": timings[max(0, math.ceil(len(timings) * .95) - 1)] if timings else None,
        "warnings": sum(bool(row["warnings"]) for row in rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--reranker")
    parser.add_argument("--rerank-minimum", type=float)
    parser.add_argument("--rerank-budget", type=int, default=40)
    parser.add_argument("--calibration")
    parser.add_argument("--split", choices=["all", "development", "heldout"], default="development")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--report", required=True)
    parser.add_argument("--distractors", help="Optional document folder to make candidate recall meaningful at scale")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="ferret-benchmark-") as tmp:
        root = Path(tmp).resolve()
        queries = materialize(root / "corpus")
        selected = [q for q in queries if args.split == "all" or q["split"] == args.split]
        if args.limit:
            selected = selected[:args.limit]
        db = str(root / "benchmark.db")
        started = time.perf_counter()
        with redirect_stdout(sys.stderr):
            init_db(db, args.model)
            paths = list((root / "corpus").iterdir())
            if args.distractors:
                paths.extend(path for path in Path(args.distractors).expanduser().rglob("*")
                             if path.is_file() and path.suffix.lower() in {".txt", ".md", ".pdf", ".docx"})
            for path in paths:
                index_file(path, db, args.model, raise_on_error=True)
        indexing_seconds = time.perf_counter() - started
        rows = []
        for number, question in enumerate(selected, 1):
            warnings = []
            ranked = []
            with redirect_stdout(sys.stderr):
                begin = time.perf_counter()
                results = search(question["query"], db, 10, args.model, reranker_path=args.reranker,
                                 rerank_minimum=args.rerank_minimum, rerank_budget=args.rerank_budget,
                                 calibration_path=args.calibration,
                                 on_ranked=ranked.extend,
                                 diagnostics=warnings)
                elapsed = (time.perf_counter() - begin) * 1000
                candidates = search(question["query"], db, 100, args.model, diagnostics=warnings)
            expected = {str(root / "corpus" / name) for name in question["relevant"]}
            ranks = [i for i, result in enumerate(results, 1) if result["path"] in expected]
            row = dict(question, rank=min(ranks) if ranks else None,
                       candidate_hit=any(r["path"] in expected for r in candidates),
                       results=[dict(filename=r["filename"], rerank_score=r.get("rerank_score")) for r in ranked],
                       elapsed_ms=round(elapsed, 1), warnings=warnings)
            rows.append(row)
            print(f"{number}/{len(selected)} {question['language']}: rank={row['rank']} {elapsed:.0f}ms", file=sys.stderr, flush=True)
        grouped = defaultdict(list)
        for row in rows:
            grouped[row["language"]].append(row)
        report = dict(model=model_spec(args.model).__dict__, reranker=args.reranker,
                      embedding_fingerprint=model_spec(args.model).fingerprint,
                      reranker_fingerprint=model_spec(args.reranker).fingerprint if args.reranker else None,
                      ranking_version=RANKING_VERSION,
                      rerank_minimum=args.rerank_minimum, split=args.split, documents=len(paths),
                      calibration=args.calibration,
                      indexing_seconds=indexing_seconds,
                      caveat="Synthetic smoke benchmark with shared scenario translations. Human relevance review and a larger corpus are required for production quality claims. Candidate hit@100 is trivial with fewer than 100 documents.",
                      overall=summarize(rows), by_language={k: summarize(v) for k, v in grouped.items()}, rows=rows)
        from core.maintenance import atomic_json
        atomic_json(args.report, report)
        print(json.dumps(report["overall"], indent=2))


if __name__ == "__main__":
    main()
