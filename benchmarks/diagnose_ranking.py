"""Read-only challenge diagnosis and independent corpus ranking ablations.

Reports stay under ignored data/, not in the indexed document corpus. No query
expansion, vocabulary overrides, or document-specific boosts are used.
"""
import argparse
from collections import defaultdict
from contextlib import redirect_stdout
import json
from pathlib import Path
import statistics
import sys
import tempfile
import time
from unittest.mock import patch

from benchmarks.corpus import materialize
from core import searcher
from core.indexer import init_db, index_file
from core.maintenance import atomic_json
from core.models import model_spec, RANKING_VERSION

CHALLENGE = [
    ("استقالات بسبب الاحتراق الوظيفي رغم تحسين مزايا المكتب", "014"),
    ("انتحال هوية مورد لتغيير وجهة الحوالة", "027"),
    ("انقطاع سلسلة التبريد أثناء النقل رغم سلامة التغليف", "038"),
    ("حجز مزدوج بسبب طلبين متزامنين", "046"),
    ("خصم مكرر بعد إعادة المحاولة بسبب مشكلة في الاتصال", "059"),
    ("تصميم يستبعد المصابين بعمى الألوان", "063"),
    ("هدر المخزون لأن الأحدث يُباع قبل الأقدم", "072"),
    ("مؤشر أداء يشجع إغلاق الشكاوى دون حلها", "085"),
    ("المزامنة ليست بديلاً عن النسخ الاحتياطي", "091"),
    ("الاستمرار في مشروع فاشل بسبب التكاليف الغارقة", "096"),
    ("الهدايا والقهوة مش معوضين الناس عن الشغل اللي مابيخلصش", "014"),
    ("Deleting a folder also erased the supposedly safe copy", "091"),
    ("2047", "059"),
]

FAMILIES = {"filename": "name", "filename_normalized": "name",
            "fts": "lexical", "normalized": "lexical",
            "semantic": "meaning", "document": "meaning", "doctype": "doctype"}
ORIGINAL = searcher._fuse_ranked_results


def family_fusion(routes, top_k):
    results = ORIGINAL(routes, sum(len(values) for _, values in routes))
    contributions = defaultdict(dict)
    for name, values in routes:
        for rank, result in enumerate(values, 1):
            family = FAMILIES[name]
            value = searcher._ROUTE_WEIGHTS[name] / (searcher._RRF_K + rank)
            contributions[result["path"]][family] = max(value, contributions[result["path"]].get(family, 0))
    for result in results:
        result["score"] = sum(contributions[result["path"]].values())
    results.sort(key=lambda r: (-r["score"], r["path"]))
    if results:
        best = results[0]["score"]
        for result in results:
            result["score"] = round(result["score"] / best, 4)
    return results[:top_k]


def coverage_fusion(routes, top_k, use_coverage=True):
    results = ORIGINAL(routes, sum(len(values) for _, values in routes))
    contributions = defaultdict(float)
    for name, values in routes:
        for rank, result in enumerate(values, 1):
            contributions[result["path"]] += (searcher._ROUTE_WEIGHTS[name]
                * (result.get("query_coverage", 1.0) if use_coverage else 1.0)
                / (searcher._RRF_K + rank))
    for result in results:
        result["score"] = contributions[result["path"]]
    results.sort(key=lambda r: (-r["score"], r["path"]))
    if results and results[0]["score"]:
        best = results[0]["score"]
        for result in results:
            result["score"] = round(result["score"] / best, 4)
    return results[:top_k]


def legacy_fusion(routes, top_k):
    return coverage_fusion(routes, top_k, use_coverage=False)


def evaluate(questions, db, model):
    rows = []
    for number, question in enumerate(questions, 1):
        expected = set(question["relevant"])
        for variant, fusion in (("baseline", legacy_fusion), ("coverage", ORIGINAL)):
            captured, ranked, warnings = {}, [], []

            def observe(routes, top_k):
                for name, values in routes:
                    captured[name] = dict(
                        rank=next((i for i, r in enumerate(values, 1) if r["filename"] in expected), None),
                        top=[r["filename"] for r in values[:3]],
                        top_coverage=[r.get("query_coverage") for r in values[:3]])
                return fusion(routes, top_k)

            start = time.perf_counter()
            with patch.object(searcher, "_fuse_ranked_results", side_effect=observe):
                searcher.search(question["query"], db, 8, model,
                                on_ranked=ranked.extend, diagnostics=warnings)
            rank = next((i for i, r in enumerate(ranked, 1) if r["filename"] in expected), None)
            rows.append(dict(question, variant=variant, rank=rank,
                             top=[r["filename"] for r in ranked[:5]], routes=captured,
                             milliseconds=round((time.perf_counter()-start)*1000, 1), warnings=warnings))
        print(f"{number}/{len(questions)} {question['group']}: "
              f"{rows[-2]['rank']} -> {rows[-1]['rank']}", file=sys.stderr, flush=True)
    return rows


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        if row["relevant"]:
            groups[(row["group"], row["variant"])].append(row)
    return {f"{group}/{variant}": dict(
        queries=len(values), hit1=sum(r["rank"] == 1 for r in values)/len(values),
        hit5=sum(r["rank"] is not None and r["rank"] <= 5 for r in values)/len(values),
        mrr10=statistics.mean(1/r["rank"] if r["rank"] and r["rank"] <= 10 else 0 for r in values),
        median_ms=statistics.median(r["milliseconds"] for r in values),
        warnings=sum(bool(r["warnings"]) for r in values))
        for (group, variant), values in groups.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/settings.json")
    parser.add_argument("--report", default="data/evaluations/ranking-ablation.json")
    parser.add_argument("--validation", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    folder = Path.home() / "Downloads/ferret-arabic-challenge"
    questions = [dict(query=q + (f' in:"{folder}"' if scoped else ""),
                      relevant=[f"ورقة-{number}.txt"], group="scoped" if scoped else "library")
                 for scoped in (True, False) for q, number in CHALLENGE]
    rows = evaluate(questions, config["db_path"], config["model_path"])
    if args.validation:
        with tempfile.TemporaryDirectory(prefix="ferret-ranking-validation-") as tmp:
            root = Path(tmp)
            questions = materialize(root / "corpus")
            db = str(root / "index.db")
            with redirect_stdout(sys.stderr):
                init_db(db, config["model_path"])
                for path in sorted((root / "corpus").iterdir()):
                    index_file(path, db, config["model_path"], raise_on_error=True)
            for question in questions:
                question["group"] = question["split"]
            rows.extend(evaluate(questions, db, config["model_path"]))
    summary = summarize(rows)
    atomic_json(args.report, dict(summary=summary, rows=rows,
        embedding_fingerprint=model_spec(config["model_path"]).fingerprint,
        ranking_version=RANKING_VERSION, reranking=False,
        caveat="Synthetic relevance labels; exploratory ablation, not a human-reviewed production benchmark. Variant timings are cache-order biased."))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
