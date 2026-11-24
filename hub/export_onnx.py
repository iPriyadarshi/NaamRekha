"""Export the demo's NER models to int8 ONNX for the browser demo (hub/space/).

For each model: export the pinned PyTorch checkpoint to ONNX with Optimum, quantize
it to dynamic int8 with ONNX Runtime, and upload it as onnx/model_quantized.onnx,
the file transformers.js loads with dtype "q8". The PyTorch weights are untouched.

Needs a Hugging Face login with write access to priyadarshi-su. Run from the repo root:
  uv run --no-project --with optimum-onnx==0.1.0 --with onnxruntime==1.31.1 \
      python hub/export_onnx.py
"""

from __future__ import annotations

import os
import sys
import tempfile

from huggingface_hub import HfApi
from onnxruntime.quantization import QuantType, quantize_dynamic
from optimum.exporters.onnx import main_export

# Hub repo -> PyTorch revision to export (the same checkpoints the demo lists).
MODELS = {
    "priyadarshi-su/naamrekha-muril-hi": "ae43085acd72527eb5f0e218109e29fdfb6e282d",
    "priyadarshi-su/naamrekha-muril_mr_n300000_e3_lr2e-05_s42_36701790": "25c48dfb02801f392faebcc0c0007cbd72a5cc73",
    "priyadarshi-su/naamrekha-muril_bn_n300000_e3_lr2e-05_s42_46e8e677": "f6171a437c97e61727b3c8710e4e4b4c31514a97",
}


def main() -> None:
    api = HfApi()
    only = set(sys.argv[1:])
    for repo, revision in MODELS.items():
        if only and repo not in only:
            continue
        with tempfile.TemporaryDirectory() as tmp:
            print(f"exporting {repo}@{revision[:7]}")
            main_export(repo, output=tmp, task="token-classification", revision=revision)
            fp32 = os.path.join(tmp, "model.onnx")
            q8 = os.path.join(tmp, "model_quantized.onnx")
            quantize_dynamic(fp32, q8, weight_type=QuantType.QInt8)
            print(f"  fp32 {os.path.getsize(fp32) / 1e6:.0f} MB -> int8 {os.path.getsize(q8) / 1e6:.0f} MB")
            info = api.upload_file(
                path_or_fileobj=q8,
                path_in_repo="onnx/model_quantized.onnx",
                repo_id=repo,
                commit_message=f"Add int8 ONNX export of {revision[:7]} for transformers.js",
            )
            print(f"  uploaded {info.commit_url}")


if __name__ == "__main__":
    main()
