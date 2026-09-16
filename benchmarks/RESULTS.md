# Local retrieval validation — 2026-09-16

These are small, synthetic regression measurements, not production-quality or
comprehensive Arabic-language guarantees. Raw reports are generated locally under
ignored `data/evaluations/` or `benchmarks/results/`; they are not shipped in Git.

## Query-coverage ranking comparison

The local Qwen3-Embedding-0.6B int8 pipeline combines lexical, filename, passage,
and document retrieval. The updated fusion discounts partial lexical matches by
query-term coverage. These comparisons ran without cross-encoder reranking.

| Split | Answerable queries | Previous Hit@5 | Coverage-aware Hit@5 |
| --- | ---: | ---: | ---: |
| Development | 144 | 91.7% (132/144) | 97.2% (140/144) |
| Held-out regression | 48 | 93.8% (45/48) | 97.9% (47/48) |

The corpus contains 48 synthetic documents across 24 bilingual scenarios, with
formal Arabic, Egyptian Arabic, English, and cross-language queries. Translations
and related queries stay in the same split. Queries are not independent intents;
the corpus has not received independent human relevance review. The held-out set
has been used for regression checks, so it is not a fresh final evaluation set.

The ranking diagnostic is `python -m benchmarks.diagnose_ranking --validation`.
It also probes the configured local library and Downloads challenge folder;
see its source for fixture requirements. Reports can contain local filenames
and must remain private. For a standalone synthetic benchmark, use
`benchmarks.run` as documented in
[the evaluation guide](../docs/SEARCH_UPGRADE.md#evaluation-and-calibration).
Reports record model/runtime fingerprints; compare only compatible configurations.

## Limitations and tradeoffs

- Cross-encoder reranking is optional: its CPU cost can make interactive search
  noticeably slower. Quality and latency must be measured together.
- Earlier reranked validation reached 47/48 held-out top-five hits, but returned
  false positives on all four no-answer queries. Development-calibrated rejection
  reduced this to two of four while lowering top-five hits to 46/48. This sample
  is too small to establish reliable rejection; calibration is not enabled by default.
- Raw SQLite KNN scale tests exclude embedding inference, extraction, and UI latency;
  they are not end-to-end performance claims.
- Broader human-labeled dialect/no-answer evaluation, complex PDF/OCR fixtures,
  isolated concurrent-load measurements, and packaged desktop testing remain needed.

Source tests cover normalization, ranking, indexing, concurrency, and migrations.
Model/OCR integration tests require the corresponding local models and Tesseract
language data. See the evaluation guide for commands and dependencies.
