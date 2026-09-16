"""Headless local search: python cli.py 'annual revenue type:pdf' --json."""

import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sqlite3
import sys
import time


def main(argv=None):
    parser = argparse.ArgumentParser(description="Ferret — private, local hybrid search")
    parser.add_argument("query", nargs="?", help='Query, with optional type:pdf and in:"/folder" filters')
    parser.add_argument("--db", default="~/ferret/ferret.db", help="Existing Ferret database")
    parser.add_argument("--model", default="~/ferret/models/bge-small-en")
    parser.add_argument("--mode", choices=("hybrid", "keyword", "semantic"), default="hybrid")
    parser.add_argument("-k", "--limit", type=int, default=8)
    parser.add_argument("--json", action="store_true", help="Machine-readable output")
    parser.add_argument("--stats", action="store_true", help="Show index health instead of searching")
    parser.add_argument("--config", help="Read app settings; explicit CLI options override them")
    parser.add_argument("--reranker", help="Local reranker model directory")
    parser.add_argument("--calibration", help="Model-specific relevance calibration JSON")
    args = parser.parse_args(argv)
    if args.config:
        config = json.loads(Path(args.config).expanduser().read_text())
        from core.models import configured_reranker
        config["reranker_path"] = configured_reranker(config)
        supplied = argv if argv is not None else sys.argv[1:]
        for option, attribute, key in (("--db", "db", "db_path"), ("--model", "model", "model_path"),
                                       ("--reranker", "reranker", "reranker_path"),
                                       ("--calibration", "calibration", "calibration_path")):
            if not any(value == option or value.startswith(option + "=") for value in supplied):
                setattr(args, attribute, config.get(key, getattr(args, attribute)))
    if not args.stats and not args.query:
        parser.error("provide a query or --stats")
    if not 1 <= args.limit <= 100:
        parser.error("--limit must be between 1 and 100")
    db_path = Path(args.db).expanduser().resolve()
    if not db_path.is_file():
        parser.error(f"index does not exist: {db_path}; index folders in the Ferret app first")
    try:
        if args.stats:
            db = sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True)
            try:
                payload = {
                    "database": str(db_path),
                    "files_by_status": dict(db.execute("SELECT status, count(*) FROM files GROUP BY status")),
                    "chunks": db.execute("SELECT count(*) FROM chunks").fetchone()[0],
                    "database_bytes": db_path.stat().st_size,
                }
            finally:
                db.close()
        else:
            # Model/backend diagnostics must not contaminate JSON stdout.
            with redirect_stdout(sys.stderr):
                from core.searcher import search
                started = time.perf_counter()
                diagnostics = []
                results = search(args.query, str(db_path), args.limit, args.model, mode=args.mode,
                                 reranker_path=args.reranker, calibration_path=args.calibration, diagnostics=diagnostics)
            payload = {"query": args.query, "mode": args.mode,
                       "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
                       "results": results, "warnings": diagnostics,
                       "calibration_requested": bool(args.calibration)}
    except (sqlite3.Error, ImportError, ValueError) as exc:
        print(f"ferret: {exc}", file=sys.stderr)
        return 1
    if args.json or args.stats:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"{len(results)} results · {payload['elapsed_ms']} ms")
        for result in results:
            print(f"\n{result['filename']}  [{', '.join(result['matched_by'])}]")
            print(f"  {result['path']}\n  {result['snippet']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
