"""Measure sqlite-vec retrieval separately from model inference on synthetic vectors."""
import argparse
import json
import platform
from pathlib import Path
import statistics
import tempfile
import time

import numpy as np

from core.indexer import init_db, _connect, _serialize_vector
from core.maintenance import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1000, 10000, 50000])
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    if any(size < 100 or size > 1000000 for size in args.sizes):
        parser.error("sizes must be 100..1000000")
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for size in args.sizes:
            path = str(Path(tmp) / f"scale-{size}.db")
            init_db(path)
            db = _connect(path)
            rng = np.random.default_rng(42)
            started = time.perf_counter()
            first = None
            for offset in range(0, size, 1000):
                vectors = rng.standard_normal((min(1000, size-offset), 384)).astype(np.float32)
                vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
                first = _serialize_vector(vectors[0]) if first is None else first
                db.executemany("INSERT INTO vec_chunks(chunk_id,embedding) VALUES (?,?)",
                               ((offset+i+1, _serialize_vector(vector)) for i, vector in enumerate(vectors)))
            db.commit()
            build_seconds = time.perf_counter() - started
            times = []
            for _ in range(31):
                started = time.perf_counter()
                found = db.execute("SELECT chunk_id,distance FROM vec_chunks WHERE embedding MATCH ? AND k=100",
                                   (first,)).fetchall()
                times.append((time.perf_counter() - started) * 1000)
                if found[0][0] != 1 or abs(found[0][1]) > 1e-5:
                    raise AssertionError("Exact nearest-neighbor result was incorrect")
            db.close()
            timings = sorted(times[1:])
            results.append(dict(vectors=size, dimensions=384, build_seconds=round(build_seconds, 2),
                                p50_ms=round(statistics.median(timings), 2),
                                p95_ms=round(timings[28], 2), database_bytes=Path(path).stat().st_size))
    report = dict(platform=platform.platform(), results=results,
                  scope="Synthetic unfiltered KNN only; excludes embedding, reranking, filesystem checks and concurrent writes.")
    atomic_json(args.report, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
