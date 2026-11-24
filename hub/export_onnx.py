"""Export the demo's NER models to ONNX for the browser demo (hub/space/).

For each model: export the pinned PyTorch checkpoint to full-precision ONNX with
Optimum, score the export on its gold test set, and upload it as onnx/model.onnx
(the file transformers.js loads with dtype "fp32") only if the score matches the
logged result. The PyTorch weights are untouched. Quantized exports are not used:
dynamic int8 quantization made MuRIL predict O for every word.

Needs a Hugging Face login with write access to priyadarshi-su. Run from the repo root:
  uv run --no-project --with optimum-onnx==0.1.0 --with onnxruntime==1.31.1 \
      --with seqeval python hub/export_onnx.py [repo ...]
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import onnxruntime as ort
from huggingface_hub import HfApi
from optimum.exporters.onnx import main_export
from transformers import AutoTokenizer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from src.data import LABELS, load_split  # noqa: E402
from src.eval import evaluate  # noqa: E402

# Hub repo -> (PyTorch revision to export, gold test language, logged gold F1).
MODELS = {
    "priyadarshi-su/naamrekha-muril-hi": ("ae43085acd72527eb5f0e218109e29fdfb6e282d", "hi", 0.8274),
    "priyadarshi-su/naamrekha-muril_mr_n300000_e3_lr2e-05_s42_36701790": (
        "25c48dfb02801f392faebcc0c0007cbd72a5cc73", "mr", 0.8156,
    ),
    "priyadarshi-su/naamrekha-muril_bn_n300000_e3_lr2e-05_s42_46e8e677": (
        "f6171a437c97e61727b3c8710e4e4b4c31514a97", "bn", 0.8136,
    ),
}
TOLERANCE = 0.001


def gold_f1(onnx_path: str, repo: str, revision: str, lang: str) -> float:
    """Entity F1 of the ONNX model on the gold test set, labels on first subwords."""
    tok = AutoTokenizer.from_pretrained(repo, revision=revision)
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    inputs = {i.name for i in sess.get_inputs()}
    rows = load_split(lang, "test")
    preds = []
    for b in range(0, len(rows), 32):
        words = [w for w, _ in rows[b : b + 32]]
        enc = tok(words, is_split_into_words=True, truncation=True, max_length=128, padding=True, return_tensors="np")
        logits = sess.run(None, {k: enc[k].astype(np.int64) for k in inputs})[0]
        for i, ws in enumerate(words):
            labels, seen = ["O"] * len(ws), set()
            for j, wid in enumerate(enc.word_ids(i)):
                if wid is not None and wid not in seen:
                    seen.add(wid)
                    labels[wid] = LABELS[int(logits[i, j].argmax())]
            preds.append(labels)
    return evaluate([t for _, t in rows], preds)["F1_overall"]


def main() -> None:
    api = HfApi()
    only = set(sys.argv[1:])
    for repo, (revision, lang, logged) in MODELS.items():
        if only and repo not in only:
            continue
        with tempfile.TemporaryDirectory() as tmp:
            print(f"exporting {repo}@{revision[:7]}", flush=True)
            main_export(repo, output=tmp, task="token-classification", revision=revision)
            path = os.path.join(tmp, "model.onnx")
            f1 = gold_f1(path, repo, revision, lang)
            print(f"  {os.path.getsize(path) / 1e6:.0f} MB, {lang} gold F1 {f1:.4f} (logged {logged:.4f})", flush=True)
            if abs(f1 - logged) > TOLERANCE:
                raise SystemExit(f"  export does not reproduce the logged score; not uploading {repo}")
            info = api.upload_file(
                path_or_fileobj=path,
                path_in_repo="onnx/model.onnx",
                repo_id=repo,
                commit_message=f"Add ONNX export of {revision[:7]} for transformers.js (gold F1 {f1:.4f})",
            )
            print(f"  uploaded {info.commit_url}", flush=True)
            if api.file_exists(repo, "onnx/model_quantized.onnx"):
                api.delete_file("onnx/model_quantized.onnx", repo, commit_message="Remove broken int8 ONNX export")
                print("  removed onnx/model_quantized.onnx", flush=True)


if __name__ == "__main__":
    main()
