"""Naamapadam loader and the shared label map."""

from __future__ import annotations

import json
import os
import zipfile
from collections import Counter

from huggingface_hub import hf_hub_download

REPO_ID = "ai4bharat/naamapadam"

# Fixed label order; ids are the list indices.
LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
label2id = {label: i for i, label in enumerate(LABELS)}
id2label = {i: label for i, label in enumerate(LABELS)}

SPLITS = ("train", "val", "test")

# (tokens, tags), equal length.
Sentence = tuple[list[str], list[str]]


def _ensure_extracted(lang: str) -> str:
    """Download the language zip (cached) and extract it once; return the dir."""
    zip_path = hf_hub_download(
        REPO_ID, f"data/{lang}_IndicNER_v1.0.zip", repo_type="dataset"
    )
    out_dir = zip_path + ".extracted"
    if not os.path.isdir(out_dir):
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(out_dir)
    return out_dir


def _find_file(out_dir: str, lang: str, split: str) -> str:
    """Return the path of {lang}_{split}.json under out_dir."""
    target = f"{lang}_{split}.json"
    for root, _, files in os.walk(out_dir):
        if target in files:
            return os.path.join(root, target)
    raise FileNotFoundError(f"{target} not found under {out_dir}")


def load_split(lang: str, split: str) -> list[Sentence]:
    """Return one split as (words, tags) pairs. lang e.g. "hi"/"mr"/"bn",
    split one of SPLITS. Empty sentences dropped; tags are strings."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")

    path = _find_file(_ensure_extracted(lang), lang, split)
    rows: list[Sentence] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:  # skip blank lines in the file
                continue
            obj = json.loads(line)
            words, tags = obj["words"], obj["ner"]
            if not words:  # drop zero-length sentences
                continue
            rows.append((words, tags))
    return rows


def tag_distribution(rows: list[Sentence]) -> Counter:
    """Count tag occurrences across the given rows."""
    counts: Counter = Counter()
    for _, tags in rows:
        counts.update(tags)
    return counts
