"""Phase 4 gate check. Uses only the shared code in src/ and committed files.

Gate (docs/01-roadmap.md): the transfer matrix is filled (MuRIL trained on hi,
tested on hi/mr/bn) plus in-language runs for mr and bn.

This script checks that those rows exist for all three seeds, come from one
clean execution of the Phase 4 notebook, use the pinned Phase 3 checkpoints and
recipe, and are reported in the docs. It does not retrain or re-score any model.

Run from the repo root:  uv run python scripts/phase4_gate.py
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
from collections import Counter
from statistics import mean, stdev

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src import COLUMNS, RESULTS_PATH  # noqa: E402

MODEL = "MuRIL"
SOURCE = "hi"
TARGETS = ["mr", "bn"]
NOTEBOOK = os.path.join(REPO_ROOT, "notebooks", "phase4-transfer.ipynb")
PHASE_DOC = os.path.join(REPO_ROOT, "docs", "phase4", "README.md")
SCORE_COLS = ["F1_PER", "F1_ORG", "F1_LOC", "F1_overall"]
REQUIRED = [(SOURCE, SOURCE), *[(SOURCE, t) for t in TARGETS], *[(t, t) for t in TARGETS]]
HI_MODELS = {
    "42": ("priyadarshi-su/naamrekha-muril_hi_n300000_e3_lr2e-05_s42_bfd254ed",
           "de5204680ed4a8ab1ac842847df6866a6e2c9251", 0.8274137068534269),
    "43": ("priyadarshi-su/naamrekha-muril_hi_n300000_e3_lr2e-05_s43_223b5780",
           "f15d615b7d760d23af6c5aa111675f0d6be39c3d", 0.824173091270828),
    "44": ("priyadarshi-su/naamrekha-muril_hi_n300000_e3_lr2e-05_s44_618532f6",
           "aa2f28136b49859a460217190b85ebedfc625e32", 0.8174346201743462),
}
SEEDS = list(HI_MODELS)
HI_REPRO_TOLERANCE = 0.001  # about two entities on the 1,929-entity Hindi gold set
SUBSET_SIZE = "300000"
EPOCHS = "3"
CHECKPOINT_URL = re.compile(r"https://huggingface\.co/[^/\s]+/[^/\s]+/tree/[0-9a-f]{40}")


def check(cond: bool, msg: str) -> None:
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        raise SystemExit(f"Gate failed: {msg}")


def parse_notes(notes: str) -> dict[str, str]:
    return dict(part.split("=", 1) for part in notes.split("; ") if "=" in part)


def notebook_logged_rows(nb: dict) -> list[tuple[str, ...]]:
    """Rows the notebook printed after log_result (lines starting with MODEL,)."""
    lines = []
    for cell in nb["cells"]:
        text = "".join("".join(out.get("text", [])) for out in cell.get("outputs", [])
                       if out.get("output_type") == "stream" and out.get("name") == "stdout")
        lines.extend(l for l in text.splitlines() if l.startswith(MODEL + ","))
    return [tuple(row) for row in csv.reader(lines)]


def main() -> None:
    print("1. Results sheet")
    check(os.path.exists(RESULTS_PATH), f"{os.path.relpath(RESULTS_PATH, REPO_ROOT)} exists")
    with open(RESULTS_PATH, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = list(reader)
    check(header == COLUMNS, "header matches the fixed columns")
    phase4 = [r for r in rows if r["model"] == MODEL and "phase=4" in r["notes"]]
    check(not any("DEBUG" in r["notes"] for r in phase4), "no DEBUG rows committed")
    check(len(phase4) == len(REQUIRED) * len(SEEDS), f"{len(REQUIRED) * len(SEEDS)} Phase 4 rows logged ({len(phase4)} found)")

    print("\n2. Transfer matrix + in-language runs, per seed")
    cells: dict[tuple[str, str, str], dict[str, str]] = {}
    for seed in SEEDS:
        for train, test in REQUIRED:
            name = f"seed {seed} {train}->{test}"
            found = [r for r in phase4 if r["train_lang"] == train and r["test_lang"] == test
                     and parse_notes(r["notes"]).get("seed") == seed]
            check(len(found) == 1, f"exactly one {name} row logged ({len(found)} found)")
            r = found[0]
            for c in SCORE_COLS:
                check(0.0 < float(r[c]) <= 1.0, f"{name} {c} = {float(r[c]):.4f} is a valid entity F1")
            check(r["subset_size"] == SUBSET_SIZE and r["epochs"] == EPOCHS, f"{name} uses n=300,000 and 3 epochs")
            check(bool(CHECKPOINT_URL.fullmatch(r["checkpoint_path"])), f"{name} checkpoint_path is a pinned Hugging Face link")
            if train == SOURCE:
                repo, revision, _ = HI_MODELS[seed]
                check(r["checkpoint_path"] == f"https://huggingface.co/{repo}/tree/{revision}",
                      f"{name} uses that seed's pinned Phase 3 checkpoint")
            else:
                notes = parse_notes(r["notes"])
                check(notes.get("gpus") == "1" and notes.get("effective_bs") == "32" and int(notes.get("warmup_steps", 0)) > 0,
                      f"{name} used one GPU, effective batch 32, and warmup")
            cells[(seed, train, test)] = r
        expected = HI_MODELS[seed][2]
        measured = float(cells[(seed, SOURCE, SOURCE)]["F1_overall"])
        check(abs(measured - expected) <= HI_REPRO_TOLERANCE,
              f"seed {seed} hi->hi {measured:.4f} is within {HI_REPRO_TOLERANCE} of Phase 3 score {expected:.4f}")
    commits = {parse_notes(r["notes"]).get("git") for r in phase4}
    check(len(commits) == 1 and None not in commits, f"all Phase 4 rows come from one code commit ({sorted(map(str, commits))})")

    print("\n3. One clean Phase 4 notebook execution")
    check(os.path.exists(NOTEBOOK), f"{os.path.relpath(NOTEBOOK, REPO_ROOT)} exists")
    with open(NOTEBOOK, encoding="utf-8") as f:
        nb = json.load(f)
    counts = [c.get("execution_count") for c in nb["cells"] if c["cell_type"] == "code"]
    check(counts == list(range(1, len(counts) + 1)), "every code cell ran once, in order, from a fresh kernel")
    logged = [tuple(r[c] for c in COLUMNS) for r in phase4]
    check(Counter(notebook_logged_rows(nb)) == Counter(logged), "results rows exactly match the notebook's printed rows")

    print("\n4. Phase 4 doc")
    check(os.path.exists(PHASE_DOC), f"{os.path.relpath(PHASE_DOC, REPO_ROOT)} exists")
    with open(PHASE_DOC, encoding="utf-8") as f:
        doc = f.read()
    for train, test in REQUIRED:
        f1 = f"{mean(float(cells[(s, train, test)]['F1_overall']) for s in SEEDS):.4f}"
        check(f1 in doc, f"doc reports {train}->{test} mean F1 {f1}")

    print("\n  transfer gap (in-language - zero-shot, F1_overall), paired by seed:")
    for t in TARGETS:
        gaps = [float(cells[(s, t, t)]["F1_overall"]) - float(cells[(s, SOURCE, t)]["F1_overall"]) for s in SEEDS]
        per_seed = ", ".join(f"{s}: {g:+.4f}" for s, g in zip(SEEDS, gaps))
        print(f"    {SOURCE} -> {t}: mean {mean(gaps):+.4f}, sd {stdev(gaps):.4f} ({per_seed})")
    print("\nPhase 4 gate: PASSED")


if __name__ == "__main__":
    main()
