"""Fit a reranker threshold on development data only and expose its tradeoffs."""
import argparse
import hashlib
import json
from pathlib import Path

from core.maintenance import atomic_json


def fit(report, minimum_hit=.85):
    if report["split"] != "development" or not report.get("reranker"):
        raise ValueError("Calibration requires a reranked development-only report")
    if any(row["split"] != "development" for row in report["rows"]):
        raise ValueError("Held-out queries must not be used for calibration")
    if report["overall"]["warnings"]:
        raise ValueError("Cannot calibrate a run with runtime failures")
    positives = [r for r in report["rows"] if r["relevant"]]
    negatives = [r for r in report["rows"] if not r["relevant"]]
    if not positives or not negatives:
        raise ValueError("Calibration needs both answerable and no-answer queries")
    scores = sorted({r["rerank_score"] for row in report["rows"] for r in row["results"]
                     if r.get("rerank_score") is not None})
    if not scores:
        raise ValueError("Calibration requires scored reranker results")
    # Short queries intentionally bypass reranking and cannot be rejected by
    # a logit threshold. Include that limitation in measured false positives.
    def retained(result, threshold):
        return result.get("rerank_score") is None or result["rerank_score"] >= threshold
    thresholds = [scores[0] - 1] + [(a + b) / 2 for a, b in zip(scores, scores[1:])] + [scores[-1] + 1]
    tradeoffs = []
    for threshold in thresholds:
        hit = sum(any(r["filename"] in row["relevant"]
                      for r in [r for r in row["results"] if retained(r, threshold)][:5])
                  for row in positives) / len(positives)
        fp = sum(any(retained(r, threshold) for r in row["results"])
                 for row in negatives) / len(negatives)
        tradeoffs.append(dict(threshold=threshold, hit_at_5=hit, false_positive_rate=fp))
    eligible = [r for r in tradeoffs if r["hit_at_5"] >= minimum_hit]
    if not eligible:
        raise ValueError("This model/ranking pipeline cannot meet the requested top-five recall even without rejection")
    chosen = min(eligible, key=lambda r: (r["false_positive_rate"], -r["hit_at_5"], r["threshold"]))
    return chosen, tradeoffs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report")
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-hit", type=float, default=.85)
    args = parser.parse_args()
    raw = Path(args.report).read_bytes()
    report = json.loads(raw)
    chosen, tradeoffs = fit(report, args.minimum_hit)
    if not report.get("embedding_fingerprint") or not report.get("reranker_fingerprint"):
        raise ValueError("Rerun the benchmark to record its exact runtime fingerprints")
    profile = dict(chosen, embedding_fingerprint=report["embedding_fingerprint"],
                   ranking_version=report.get("ranking_version"),
                   reranker_fingerprint=report["reranker_fingerprint"],
                   source_sha256=hashlib.sha256(raw).hexdigest(),
                   scope="synthetic development corpus; validate on held-out and real documents",
                   tradeoffs=tradeoffs)
    atomic_json(args.output, profile)
    print(json.dumps(chosen, indent=2))


if __name__ == "__main__":
    main()
