"""Explicit model installation and resumable side-by-side index builds."""
from contextlib import contextmanager
from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile

from core.models import ModelSpec, load_model, model_spec

PRESETS = {
    "qwen3-embedding-0.6b": ("onnx-community/Qwen3-Embedding-0.6B-ONNX", ModelSpec(
        name="Qwen/Qwen3-Embedding-0.6B", dimensions=1024, max_tokens=8192, pooling="last",
        query_prefix="Instruct: Given a search query, retrieve relevant passages that answer the query\nQuery: ",
        onnx_file="onnx/model_int8.onnx", batch_size=2)),
    "bge-m3": ("Xenova/bge-m3", ModelSpec(
        name="BAAI/bge-m3", dimensions=1024, max_tokens=8192, pooling="cls",
        query_prefix="", onnx_file="onnx/model_int8.onnx", batch_size=2)),
    "multilingual-e5-small": ("Xenova/multilingual-e5-small", ModelSpec(
        name="intfloat/multilingual-e5-small", pooling="mean", query_prefix="query: ",
        passage_prefix="passage: ", onnx_file="onnx/model_quantized.onnx")),
    "bge-reranker-v2-m3": ("onnx-community/bge-reranker-v2-m3-ONNX", ModelSpec(
        name="BAAI/bge-reranker-v2-m3", kind="reranker", query_prefix="",
        onnx_file="onnx/model_int8.onnx", batch_size=2)),
}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def config_lock(config_path):
    """Exclusive app/maintenance lock; the lock file is harmless after a crash."""
    path = Path(str(config_path) + ".lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        try:
            if os.name == "nt":
                import msvcrt
                handle.seek(0)
                handle.write(b"0")
                handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("Ferret is running or another rebuild is active. Quit it before maintenance.") from exc
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def install_model(preset, destination):
    """Only this explicitly invoked function accesses the network."""
    from huggingface_hub import HfApi, hf_hub_download
    root = Path(destination).expanduser().resolve()
    repo, template = PRESETS[preset]
    if (root / "ferret-model.json").exists():
        if model_spec(root).name != template.name:
            raise ValueError("Directory contains a different model; choose a new destination")
        load_model(root)
        return str(root)
    api = HfApi()
    revision = api.model_info(repo).sha
    # Pin all artifacts to one immutable upstream commit, including resumes.
    pending = root / "download.json"
    if pending.exists():
        record = json.loads(pending.read_text())
        if record["repo"] != repo:
            raise ValueError("Directory contains another model download")
        revision = record["revision"]
    else:
        atomic_json(pending, {"repo": repo, "revision": revision})
    for filename in ("tokenizer.json", template.onnx_file):
        print(f"Downloading {repo}@{revision[:12]} / {filename}", flush=True)
        hf_hub_download(repo, filename, revision=revision, local_dir=str(root))
    spec = ModelSpec(**dict(asdict(template), revision=revision))
    # Validate actual inference before publishing the manifest.
    from core.models import OnnxModel
    model = OnnxModel(root, spec)
    if spec.kind == "embedding":
        model.embed(["مرحبا بالعالم", "Hello world"])
    else:
        model.rerank("عمل", ["العمل في المكتب"])
    atomic_json(root / "ferret-model.json", asdict(spec))
    return str(root)


def install_ocr(destination):
    """Install Arabic/English language data locally without changing the OS."""
    from urllib.request import urlopen
    import hashlib
    root = Path(destination).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    # Resolve a single upstream commit, then pin both files to it.
    record_path = root / "source.json"
    if record_path.exists():
        record = json.loads(record_path.read_text())
        revision, checksums = record["revision"], record["sha256"]
    else:
        with urlopen("https://api.github.com/repos/tesseract-ocr/tessdata_fast/commits/main", timeout=30) as response:
            revision = json.load(response)["sha"]
        checksums = {}
        atomic_json(record_path, dict(repository="tesseract-ocr/tessdata_fast", revision=revision, sha256=checksums))
    for language in ("ara", "eng"):
        target = root / f"{language}.traineddata"
        if target.exists():
            if checksums.get(target.name) == hashlib.sha256(target.read_bytes()).hexdigest():
                continue
            raise ValueError(f"OCR file checksum does not match its download record: {target}")
        url = f"https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/{revision}/{language}.traineddata"
        with urlopen(url, timeout=120) as response:
            data = response.read()
        target.write_bytes(data)
        checksums[target.name] = hashlib.sha256(data).hexdigest()
        atomic_json(record_path, dict(repository="tesseract-ocr/tessdata_fast", revision=revision, sha256=checksums))
    return str(root)


def rebuild(config_path, model_path, output, activate=False, reranker_path=None):
    from core.indexer import init_db, index_file, _connect, _delete_file_data
    from core.reconciler import reconcile_filesystem, ACTION_MISSING, ACTION_EXCLUDED
    config_path = Path(config_path).resolve()
    output = Path(output).expanduser().resolve()
    model_path = str(Path(model_path).expanduser().resolve())
    # The active app may keep serving its old index while this builds. Only
    # final activation needs exclusive access to the configuration.
    with config_lock(output):
        config = json.loads(config_path.read_text())
        original_db = Path(config["db_path"]).expanduser().resolve()
        if output == original_db or (output.exists() and original_db.exists() and os.path.samefile(output, original_db)):
            raise ValueError("Rebuild destination must differ from the active index")
        folders = config.get("indexed_folders", [])
        if not folders or any(not Path(folder).expanduser().is_dir() for folder in folders):
            raise ValueError("Every configured source folder must be available for a complete rebuild")
        model = load_model(model_path)
        model.embed(["اختبار البحث", "search test"], is_query=True)
        if reranker_path:
            load_model(reranker_path).rerank("test", ["a test passage"])
        output.parent.mkdir(parents=True, exist_ok=True)
        init_db(str(output), model_path)
        from core.extractor import configure_ocr
        configure_ocr(config)
        exclusions = list(dict.fromkeys([*config.get("exclude_patterns", []),
                                        ".venv", "venv", ".git", "__pycache__", "node_modules", "models"]))
        # Repeat discovery to catch writes/moves/deletions during the rebuild.
        # Refuse activation if sources never become stable; the old index stays usable.
        for iteration in range(5):
            actions = reconcile_filesystem(folders, str(output), exclude_patterns=exclusions)
            if not actions:
                break
            for number, action in enumerate(actions, 1):
                print(f"Pass {iteration + 1}: {number}/{len(actions)} {action.kind}: {action.path}", flush=True)
                if action.kind in {ACTION_MISSING, ACTION_EXCLUDED}:
                    db = _connect(str(output))
                    try:
                        row = db.execute("SELECT id FROM files WHERE path=?", (action.path,)).fetchone()
                        if row:
                            _delete_file_data(db, row[0])
                            db.execute("DELETE FROM files WHERE id=?", (row[0],))
                            db.commit()
                    finally:
                        db.close()
                else:
                    if action.previous_path:
                        from core.indexer import move_indexed_file
                        move_indexed_file(action.previous_path, action.path, str(output))
                    index_file(action.path, str(output), model_path, raise_on_error=True)
        else:
            raise RuntimeError("Source folders are still changing; rerun to resume before activation")
        db = _connect(str(output))
        try:
            if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise RuntimeError("Index integrity check failed")
            chunks = db.execute("SELECT count(*) FROM chunks").fetchone()[0]
            vectors = db.execute("SELECT count(*) FROM vec_chunks").fetchone()[0]
            if chunks != vectors or not chunks:
                raise RuntimeError("Index is empty or has missing vectors")
            db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        finally:
            db.close()
        if activate:
            updated = dict(config, db_path=str(output), model_path=model_path,
                           ocr_languages=config.get("ocr_languages", "ara+eng"))
            updated.pop("rerank_minimum", None)
            updated.pop("calibration_path", None)
            if reranker_path:
                updated["reranker_path"] = str(Path(reranker_path).expanduser().resolve())
            # Backup is a full usable configuration, pointing at the retained index.
            with config_lock(config_path):
                if json.loads(config_path.read_text()) != config:
                    raise RuntimeError("Settings changed during the build; rerun to reconcile before activation")
                atomic_json(str(config_path) + ".previous", config)
                atomic_json(config_path, updated)
        return {"database": str(output), "chunks": chunks, "activated": activate}
