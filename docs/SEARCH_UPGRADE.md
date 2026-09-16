# Bilingual search upgrade

Ferret now has separate local model adapters, Arabic lexical matching, document
and passage retrieval, and optional multilingual cross-encoder reranking. All
inference is offline. Only the explicitly invoked download commands use the network.

## Install and upgrade an existing index

Run these commands from the repository with its virtual environment activated.
Use Settings first to save the folders you want indexed.

```bash
python manage.py download qwen3-embedding-0.6b models/qwen3-embedding-0.6b
python manage.py download bge-reranker-v2-m3 models/bge-reranker-v2-m3
python manage.py download-ocr models/tessdata

python manage.py rebuild --config config/settings.json \
  --model models/qwen3-embedding-0.6b \
  --reranker models/bge-reranker-v2-m3 \
  --output data/ferret-multilingual.db --activate
```

Embedding alternatives are `bge-m3` and `multilingual-e5-small`, using matching
destination directories. E5 is smaller and faster to index; choose based on the
benchmark and your own documents. Model installation resolves an immutable upstream
revision and validates actual inference before marking a download complete. It can
resume a partial download. No weights are checked into git.

The app can keep using its old index while a separate build runs. Quit the app
before activation; an exclusive configuration lock prevents switching underneath
a running new-version app. Use a distinct output path for each model/version.
If activation is blocked, quit and repeat the same command: completed files are
reused, changes are reconciled, and the build resumes. A failure leaves the old
configuration active. Missing source folders, extraction failures, empty builds,
and unstable source trees prevent activation rather than silently losing coverage.

Restart Ferret after activation. To switch back:

```bash
python manage.py rollback --config config/settings.json
```

The previous configuration and database are retained. Builds take extra disk space;
the implementation does not automatically delete old indexes or downloaded models.

## Arabic behavior

- Arabic, English, and mixed-script queries are accepted by the semantic route.
- Original text feeds the embedding model and remains the displayed evidence.
- A separate FTS index normalizes Arabic presentation forms, optional diacritics,
  tatweel, alef variants, Persian kaf/yeh, and Arabic/Persian digits.
- Conservative article variants broaden lexical recall without root stemming or
  merging ta marbuta with ha. Exact lexical evidence carries greater weight.
- Arabic filenames also support normalization. RTL previews use Qt's bidirectional
  rendering; mixed English/Arabic text is not manually reversed.
- Default configured OCR languages are `ara+eng`. The local tessdata download
  supplies both languages; the Tesseract executable must also be installed.
- PDF OCR preserves page boundaries. Empty or visibly broken native text falls
  back to OCR; failures are recorded. UTF-8/UTF-16 and DOCX tables are supported.
  Legacy text decoding is marked with a warning so its preview can be checked.

Egyptian Arabic is included in evaluation. Dialect quality, complex PDF reading
order, handwriting, and damaged scans remain model/document dependent. This is not
a guarantee of perfect Arabic understanding or OCR.

## Search and diagnostics

```bash
python cli.py 'ضغط العمل' --config config/settings.json --json
python cli.py 'employee burnout type:txt' --config config/settings.json
python cli.py 'إجازة ٢٠٢٦' --config config/settings.json --mode keyword
python cli.py --config config/settings.json --stats
```

The desktop presents initial keyword matches and then hybrid candidates. When deep
reranking is enabled, it refines those candidates afterward. New input cancels
obsolete reranking between bounded batches. Match rows
show retrieval routes and original page/line evidence. Model failures are reported
as incomplete search instead of being disguised as no matches. Settings show file
states and extraction-warning counts; indexing progress is shown for empty searches.

Desktop search and `cli.py --config` now default to fast hybrid retrieval without
the expensive cross-encoder pass. Enable **deep reranking** in Settings only when
you want that extra pass; on this library it previously took about nine seconds on
CPU. The model path is retained, and `cli.py --reranker models/bge-reranker-v2-m3`
explicitly requests it. `reranking_enabled` in settings JSON controls the default.
The multilingual embedding model and index do not change. Search inference takes
priority between indexing batches instead of waiting for a whole document.

The retrieval sequence is filters → filename/FTS/normalized FTS/passage vectors/
document vectors → rank fusion → optional passage reranking → one result per file.
Partial keyword and filename matches now contribute in proportion to query-term
coverage, so a match on one incidental word is not treated as full-query evidence.
This is a ranking-only change: no embedding rebuild is required. Existing
calibration profiles must be regenerated because the ranking contract changed.
Reranking is blended with retrieval ranks; one- and two-token queries keep the
hybrid ordering because underspecified queries benefit less from passage reranking.
Identical file copies share a result slot, with their locations in the tooltip.
New-model indexing combines focused 256-token chunks with wider source windows.
Document vectors pool all indexed passages, not just the opening paragraph. Cached
query embeddings are bounded and tied to the loaded model. Source files and queries
are never sent to a remote inference service.
Quantized decoder embeddings run one sequence at a time so padding and neighboring
documents cannot alter their vectors. This runtime contract is part of the index
fingerprint; earlier experimental Qwen indexes must be rebuilt.

Without calibration, semantic results are ranked suggestions, including for queries
with no relevant document. RRF scores and reranker logits are not probabilities.
Do not infer certainty from a first-place result or a normalized score of 1.

## Evaluation and calibration

See [measured results and remaining gaps](../benchmarks/RESULTS.md) for synthetic
regression measurements and their limitations. Generated JSON reports stay local:
`benchmarks/results/` and `data/evaluations/` are ignored by Git.

```bash
QT_QPA_PLATFORM=offscreen python -m unittest discover -s tests -p '*unit.py'
QT_QPA_PLATFORM=offscreen python -m unittest discover -s tests -p 'test_arabic_integration.py'

python -m benchmarks.run --model models/qwen3-embedding-0.6b \
  --reranker models/bge-reranker-v2-m3 --split development --rerank-budget 20 \
  --report benchmarks/results/my-development.json

python -m benchmarks.calibrate benchmarks/results/my-development.json \
  --output benchmarks/results/my-calibration.json

python -m benchmarks.run --model models/qwen3-embedding-0.6b \
  --reranker models/bge-reranker-v2-m3 --split heldout --rerank-budget 20 \
  --calibration benchmarks/results/my-calibration.json \
  --report benchmarks/results/my-heldout.json
```

The corpus contains 24 bilingual scenarios, 192 answerable queries (including scoped
cross-language variants), and 16 no-answer queries. Translations and related queries
stay in the same split. It is a **synthetic acceptance corpus**, not an independently
human-reviewed production benchmark. Candidate hit@100 is trivial in its 48-document
collection; use `--distractors /a/document/folder` to test retrieval coverage at scale.
That optional folder is read locally and indexed only in a temporary benchmark index.

Calibration refuses held-out data, rejects failed runs, records model fingerprints,
and reports the tradeoff between top-five hits and false positives. It is not applied
automatically: evaluate the tradeoff first. Use `--calibration` in the CLI, or set
`calibration_path` in app settings JSON. Changing models invalidates calibration.
One- and two-token queries bypass reranking and its rejection threshold.
The maintenance command clears old calibration settings on model/index activation.

For the existing library and difficult story, a separate read-only diagnostic is:

```bash
python -m benchmarks.probe --model models/qwen3-embedding-0.6b \
  --reranker models/bge-reranker-v2-m3 --report benchmarks/results/story.json
```

It re-embeds existing extracted passages without modifying the active index. This
can take substantially longer than the small synthetic benchmark. Its report omits
source text and unrelated filenames. A cheaper `--sample-per-file 1` run explicitly
samples one passage per file and must not be described as full-corpus verification.
Local embedding caches are stored under ignored `data/probe-cache/`.

```bash
python -m benchmarks.scale --report benchmarks/results/scale.json
```

Scale measurements isolate raw sqlite-vec KNN from model inference. An approximate
index such as zvec remains a measured future option; this release retains SQLite.
Generated summaries and local query expansion are also deferred experiments rather
than silently invented document evidence.

## Packaging

Set `FERRET_BUNDLE_MODEL=models/qwen3-embedding-0.6b` when building to bundle that
specific model. Packaging does not sweep every experimental model into the app.
Local `models/tessdata` is bundled when present; Debian dependencies include Arabic
and English language packs. Existing release artifacts are not rebuilt by this change.
