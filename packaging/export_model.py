"""
Export BAAI/bge-small-en to ONNX format using torch.onnx.export.
Output matches the structure expected by core/indexer.py:
  models/bge-small-en/onnx/model.onnx
  models/bge-small-en/tokenizer.json
"""
import sys
from pathlib import Path

import torch  # pip install torch transformers onnx onnxscript
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "BAAI/bge-small-en"
OUTPUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("models/bge-small-en")
ONNX_DIR   = OUTPUT_DIR / "onnx"
ONNX_PATH  = ONNX_DIR / "model.onnx"

ONNX_DIR.mkdir(parents=True, exist_ok=True)

print(f"Downloading {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

print("Saving tokenizer ...")
tokenizer.save_pretrained(str(OUTPUT_DIR))

print("Exporting to ONNX ...")
dummy = tokenizer(
    "export dummy input",
    return_tensors="pt",
    padding="max_length",
    max_length=512,
    truncation=True,
)

with torch.no_grad():
    torch.onnx.export(
        model,
        (dummy["input_ids"], dummy["attention_mask"], dummy["token_type_ids"]),
        str(ONNX_PATH),
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids":        {0: "batch", 1: "seq"},
            "attention_mask":   {0: "batch", 1: "seq"},
            "token_type_ids":   {0: "batch", 1: "seq"},
            "last_hidden_state":{0: "batch", 1: "seq"},
        },
        opset_version=18,
    )

print(f"Done: {ONNX_PATH}")
