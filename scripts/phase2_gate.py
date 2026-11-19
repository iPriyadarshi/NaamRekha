"""Phase 2 gate check. Uses only the shared code in src/ and committed files.

Gate (docs/01-roadmap.md): the BiLSTM-CRF Hindi gold entity F1 is logged in
the results sheet. This script checks that the row exists, follows the
results-sheet rules in docs/03-conventions.md, and matches the output saved
in the Phase 2 notebook, so the number is traceable to a real run.

It does not retrain or re-score the model (the checkpoint lives on Drive).

Run from the repo root:  uv run python scripts/phase2_gate.py
"""

from __future__ import annotations

import csv
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src import COLUMNS, RESULTS_PATH  # noqa: E402

MODEL = "BiLSTM-CRF"
LANG = "hi"
NOTEBOOK = os.path.join(REPO_ROOT, "notebooks", "phase2-bilstm-crf.ipynb")
PHASE_DOC = os.path.join(REPO_ROOT, "docs", "phase2", "README.md")
SCORE_COLS = ["F1_PER", "F1_ORG", "F1_LOC", "F1_overall"]
# Columns that must agree between results.csv and the notebook's logged row.
# checkpoint_path and notes may be edited after the run (e.g. Drive share link).
TRACE_COLS = ["model", "train_lang", "test_lang", "subset_size", "epochs", *SCORE_COLS]


def check(cond: bool, msg: str) -> None:
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        raise SystemExit(f"Gate failed: {msg}")


def notebook_logged_rows() -> list[dict[str, str]]:
    """Rows the notebook printed after log_result (lines starting with MODEL,)."""
    with open(NOTEBOOK, encoding="utf-8") as f:
        nb = json.load(f)
    rows = []
    for cell in nb["cells"]:
        for out in cell.get("outputs", []):
            for line in "".join(out.get("text", [])).splitlines():
                if line.startswith(MODEL + ","):
                    rows.append(dict(zip(COLUMNS, next(csv.reader([line])))))
    return rows


def main() -> None:
    print("1. Results sheet")
    check(os.path.exists(RESULTS_PATH), f"{os.path.relpath(RESULTS_PATH, REPO_ROOT)} exists")
    with open(RESULTS_PATH, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = list(reader)
    check(header == COLUMNS, "header matches the fixed columns")

    runs = [r for r in rows if r["model"] == MODEL and r["train_lang"] == LANG and r["test_lang"] == LANG]
    check(len(runs) >= 1, f"{MODEL} {LANG} -> {LANG} row logged ({len(runs)} found)")
    real = [r for r in runs if "DEBUG" not in r["notes"]]
    check(len(real) >= 1, "at least one non-DEBUG run")
    row = max(real, key=lambda r: float(r["F1_overall"]))

    print("\n2. Row follows docs/03-conventions.md")
    scores = {c: float(row[c]) for c in SCORE_COLS}
    for c in SCORE_COLS:
        check(0.0 < scores[c] <= 1.0, f"{c} = {scores[c]:.4f} is a valid entity F1")
    check(int(row["subset_size"]) > 0, f"subset_size = {int(row['subset_size']):,}")
    check(int(row["epochs"]) > 0, f"epochs = {row['epochs']}")
    check(bool(row["checkpoint_path"].strip()), f"checkpoint_path = {row['checkpoint_path']}")
    check("seed=" in row["notes"], "seed recorded in notes")

    print("\n3. Traceable to the Phase 2 notebook")
    check(os.path.exists(NOTEBOOK), f"{os.path.relpath(NOTEBOOK, REPO_ROOT)} exists")
    logged = notebook_logged_rows()
    check(len(logged) >= 1, "notebook output contains the logged results row")
    match = any(all(nb_row[c] == row[c] for c in TRACE_COLS) for nb_row in logged)
    check(match, "results.csv row matches the notebook output (" + ", ".join(TRACE_COLS) + ")")

    print("\n4. Phase 2 doc")
    check(os.path.exists(PHASE_DOC), f"{os.path.relpath(PHASE_DOC, REPO_ROOT)} exists")
    with open(PHASE_DOC, encoding="utf-8") as f:
        doc = f.read()
    check(f"{scores['F1_overall']:.4f}" in doc, f"doc reports the logged F1 {scores['F1_overall']:.4f}")

    print(f"\n  {MODEL} Hindi gold F1: overall {scores['F1_overall']:.4f} | "
          f"PER {scores['F1_PER']:.4f} | ORG {scores['F1_ORG']:.4f} | LOC {scores['F1_LOC']:.4f}")
    print(f"  Phase 3 bar: MuRIL must clearly beat {scores['F1_overall']:.4f}")
    print("\nPhase 2 gate: PASSED")


if __name__ == "__main__":
    main()
