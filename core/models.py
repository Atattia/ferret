"""Offline ONNX model contracts. Downloads are a separate explicit operation."""
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import threading
from collections import OrderedDict
from contextlib import contextmanager

import numpy as np

RANKING_VERSION = "hybrid-documents-coverage-rrf-v2"


def configured_reranker(config):
    """Heavy CPU reranking is explicit opt-in, not the interactive default."""
    return config.get("reranker_path") if config.get("reranking_enabled", False) else None


class InferenceGate:
    """Yield between indexing batches and prioritize waiting search queries."""
    def __init__(self):
        self.condition = threading.Condition()
        self.busy = False
        self.queries = 0

    @contextmanager
    def slot(self, query=False):
        with self.condition:
            if query:
                self.queries += 1
                self.condition.notify_all()
            try:
                self.condition.wait_for(lambda: not self.busy and (query or not self.queries))
                self.busy = True
            finally:
                if query:
                    self.queries -= 1
        try:
            yield
        finally:
            with self.condition:
                self.busy = False
                self.condition.notify_all()


@dataclass(frozen=True)
class ModelSpec:
    name: str = "BAAI/bge-small-en"
    revision: str = "legacy"
    dimensions: int = 384
    max_tokens: int = 512
    pooling: str = "cls"
    query_prefix: str = "Represent this sentence for searching relevant passages: "
    passage_prefix: str = ""
    onnx_file: str = "onnx/model.onnx"
    kind: str = "embedding"
    batch_size: int = 8
    threads: int = 2

    @property
    def fingerprint(self):
        contract = asdict(self)
        if self.pooling == "last":
            contract["embedding_runtime"] = "single-sequence-v1"
        return hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()


def model_spec(path):
    manifest = Path(path).expanduser() / "ferret-model.json"
    if not manifest.exists():
        return ModelSpec()
    spec = ModelSpec(**json.loads(manifest.read_text()))
    if (spec.pooling not in {"cls", "mean", "last", "sentence"}
            or spec.kind not in {"embedding", "reranker"}
            or not 1 <= spec.dimensions <= 8192 or not 16 <= spec.max_tokens <= 32768
            or not 1 <= spec.batch_size <= 128 or not 1 <= spec.threads <= 32):
        raise ValueError("Invalid model manifest")
    return spec


def has_manifest(path):
    return (Path(path).expanduser() / "ferret-model.json").is_file()


class OnnxModel:
    def __init__(self, path, spec):
        import onnxruntime as ort
        ort.disable_telemetry_events()
        from tokenizers import Tokenizer
        root = Path(path).expanduser().resolve()
        self.spec = spec
        self.lock = threading.RLock()
        self.inference_gate = InferenceGate()
        self.query_cache = OrderedDict()
        self.tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
        pad = self.tokenizer.token_to_id("<pad>")
        pad_token = "<pad>" if pad is not None else "[PAD]"
        pad = pad if pad is not None else self.tokenizer.token_to_id("[PAD]")
        if pad is None:
            pad_token = "<|endoftext|>"
            pad = self.tokenizer.token_to_id(pad_token)
        if pad is None:
            raise ValueError("Model tokenizer has no supported padding token")
        self.tokenizer.enable_padding(pad_id=pad, pad_token=pad_token,
                                      direction="left" if spec.pooling == "last" else "right")
        self.tokenizer.enable_truncation(max_length=spec.max_tokens)
        # Chunking needs a separate tokenizer without truncation or padding.
        self.chunk_tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
        self.chunk_tokenizer.no_truncation()
        self.chunk_tokenizer.no_padding()
        options = ort.SessionOptions()
        options.intra_op_num_threads = spec.threads
        self.session = ort.InferenceSession(str(root / spec.onnx_file), sess_options=options,
                                           providers=["CPUExecutionProvider"])
        self.input_names = {item.name for item in self.session.get_inputs()}
        self.kv_inputs = {item.name: item for item in self.session.get_inputs()
                          if item.name.startswith("past_key_values.")}
        self.output_name = self.session.get_outputs()[0].name
        unexpected = self.input_names - {"input_ids", "attention_mask", "token_type_ids", "position_ids"} - self.kv_inputs.keys()
        if unexpected:
            raise ValueError("Unsupported model inputs: " + ", ".join(sorted(unexpected)))

    def _run(self, items):
        encoded = self.tokenizer.encode_batch(items)
        inputs = {
            "input_ids": np.asarray([e.ids for e in encoded], dtype=np.int64),
            "attention_mask": np.asarray([e.attention_mask for e in encoded], dtype=np.int64),
            "token_type_ids": np.asarray([e.type_ids for e in encoded], dtype=np.int64),
        }
        inputs["position_ids"] = np.maximum(inputs["attention_mask"].cumsum(axis=1) - 1, 0).astype(np.int64)
        # Decoder-style embedding exports expose a KV cache. Embedding is one
        # full forward pass, so its past length is always zero, never shared.
        for name, metadata in self.kv_inputs.items():
            shape = metadata.shape
            if len(shape) != 4 or not isinstance(shape[1], int) or not isinstance(shape[3], int):
                raise ValueError("Unsupported embedding KV cache shape")
            dtype = np.float16 if metadata.type == "tensor(float16)" else np.float32
            inputs[name] = np.zeros((len(items), shape[1], 0, shape[3]), dtype=dtype)
        output = self.session.run([self.output_name], {k: v for k, v in inputs.items() if k in self.input_names})[0]
        return output, inputs["attention_mask"]

    def embed(self, texts, is_query=False):
        if self.spec.kind != "embedding":
            raise ValueError("A reranker cannot produce index embeddings")
        prefix = self.spec.query_prefix if is_query else self.spec.passage_prefix
        vectors = []
        with self.lock:
            if is_query and len(texts) == 1 and texts[0] in self.query_cache:
                self.query_cache.move_to_end(texts[0])
                return self.query_cache[texts[0]].copy()
        # Do not hold the cache lock across a whole document's inference.
        # Quantized decoder embeddings remain single-sequence and deterministic.
        batch_size = 1 if self.spec.pooling == "last" else self.spec.batch_size
        for start in range(0, len(texts), batch_size):
            with self.inference_gate.slot(query=is_query):
                output, mask = self._run([prefix + text for text in texts[start:start + batch_size]])
                if self.spec.pooling == "sentence":
                    pooled = output
                elif self.spec.pooling == "mean":
                    pooled = (output * mask[:, :, None]).sum(1) / mask.sum(1)[:, None].clip(min=1)
                elif self.spec.pooling == "last":
                    indices = (mask * np.arange(mask.shape[1])[None, :]).max(1)
                    pooled = output[np.arange(len(output)), indices]
                else:
                    pooled = output[:, 0, :]
                if pooled.ndim != 2 or pooled.shape[1] != self.spec.dimensions:
                    raise ValueError("Model output dimensions do not match its manifest")
                if not np.isfinite(pooled).all():
                    raise ValueError("Model produced non-finite embeddings")
                norms = np.linalg.norm(pooled, axis=1, keepdims=True)
                if not np.isfinite(norms).all() or (norms < 1e-9).any():
                    raise ValueError("Model produced zero or invalid embedding norms")
                vectors.append((pooled / norms).astype(np.float32))
        result = np.concatenate(vectors) if vectors else np.zeros((0, self.spec.dimensions), np.float32)
        with self.lock:
            if is_query and len(texts) == 1:
                self.query_cache[texts[0]] = result.copy()
                while len(self.query_cache) > 128:
                    self.query_cache.popitem(last=False)
            return result

    def rerank(self, query, passages, cancelled=None):
        if self.spec.kind != "reranker":
            raise ValueError("Select a cross-encoder reranker model")
        scores = []
        with self.lock:
            for start in range(0, len(passages), self.spec.batch_size):
                if cancelled and cancelled():
                    return []
                output, _ = self._run([(query, text) for text in passages[start:start + self.spec.batch_size]])
                if output.ndim != 2 or output.shape[1] != 1:
                    raise ValueError("Reranker must output one relevance logit per pair")
                scores.extend(float(score) for score in output[:, 0])
        if not np.isfinite(scores).all():
            raise ValueError("Reranker produced invalid scores")
        return scores


@lru_cache(maxsize=2)
def _load(path, spec):
    return OnnxModel(path, spec)


def load_model(path):
    return _load(str(Path(path).expanduser().resolve()), model_spec(path))


def check_index_model(db, path):
    row = db.execute("SELECT value FROM index_metadata WHERE key='model_fingerprint'").fetchone()
    expected = model_spec(path).fingerprint
    if row and row[0] != expected:
        raise ValueError("This index was built with another model. Rebuild into a new index before switching.")
    if not row and has_manifest(path):
        raise ValueError("Legacy index requires a separate rebuild for the multilingual model.")


def calibrated_threshold(path, embedding_path, reranker_path):
    profile = json.loads(Path(path).expanduser().read_text())
    if profile.get("ranking_version") != RANKING_VERSION:
        raise ValueError("Calibration belongs to a different ranking pipeline; recalibrate")
    if (profile["embedding_fingerprint"] != model_spec(embedding_path).fingerprint
            or profile["reranker_fingerprint"] != model_spec(reranker_path).fingerprint):
        raise ValueError("Calibration belongs to different models; rerun development calibration")
    threshold = float(profile["threshold"])
    if not np.isfinite(threshold):
        raise ValueError("Invalid calibration threshold")
    return threshold
