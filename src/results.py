"""Append one run per call to results/results.csv with the fixed columns."""

from __future__ import annotations

import csv
import os

COLUMNS = [
    "model",
    "train_lang",
    "test_lang",
    "subset_size",
    "epochs",
    "F1_PER",
    "F1_ORG",
    "F1_LOC",
    "F1_overall",
    "checkpoint_path",
    "notes",
]

# repo root is the parent of src/
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_PATH = os.path.join(_REPO_ROOT, "results", "results.csv")


def log_result(
    *,
    model: str,
    train_lang: str,
    test_lang: str,
    subset_size: int,
    epochs: int,
    scores: dict[str, float],
    checkpoint_path: str = "",
    notes: str = "",
    path: str = RESULTS_PATH,
) -> None:
    """Append one run. scores is the dict from eval.evaluate.
    Writes the header if the file is new or empty."""
    row = {
        "model": model,
        "train_lang": train_lang,
        "test_lang": test_lang,
        "subset_size": subset_size,
        "epochs": epochs,
        "F1_PER": scores.get("F1_PER", ""),
        "F1_ORG": scores.get("F1_ORG", ""),
        "F1_LOC": scores.get("F1_LOC", ""),
        "F1_overall": scores.get("F1_overall", ""),
        "checkpoint_path": checkpoint_path,
        "notes": notes,
    }

    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
