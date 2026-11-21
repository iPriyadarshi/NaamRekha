"""Phase 3 gate check. Uses only the shared code in src/ and committed files.

Gate (docs/01-roadmap.md): both MuRIL and IndicBERT are scored on Hindi gold;
MuRIL clearly beats the BiLSTM-CRF baseline (0.8038, logged in Phase 2); the
scaling curve (F1 vs subset_size) is logged in the results sheet and a
working subset size is chosen from it.

This script checks that every Phase 3 row comes from one top-to-bottom
notebook execution of one commit, uses the fixed training setup, links to a
pinned checkpoint, and agrees with the chosen size in the docs. It does not
retrain or re-score any model.

Run from the repo root:  uv run python scripts/phase3_gate.py
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict
from statistics import mean

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src import COLUMNS, RESULTS_PATH  # noqa: E402

LANG = "hi"
NOTEBOOK = os.path.join(REPO_ROOT, "notebooks", "phase3-muril-indicbert.ipynb")
PHASE_DOC = os.path.join(REPO_ROOT, "docs", "phase3", "README.md")
ROADMAP = os.path.join(REPO_ROOT, "docs", "01-roadmap.md")
DECISIONS = os.path.join(REPO_ROOT, "docs", "06-decisions.md")
SCORE_COLS = ["F1_PER", "F1_ORG", "F1_LOC", "F1_overall"]
BASELINE_MODEL = "BiLSTM-CRF"
PHASE_MODELS = ("MuRIL", "IndicBERT")
REQUIRED_NOTES = ("seed", "lr", "effective_bs", "gpus", "warmup_steps", "config", "val_F1", "git", "torch", "transformers")
MIN_SCALING_POINTS = 3  # at least this many distinct subset sizes logged for MuRIL
MIN_MARGIN = 0.01  # "clearly beats" requires at least this much headroom over baseline
CURVE_EPOCHS = "3"
CURVE_LR = "2e-05"
HANDOFF_SEED = "42"
REQUIRED_SEEDS = {"42", "43", "44"}  # the chosen size must be confirmed on all of these
SIZE_TOLERANCE = 0.01  # same rule as the notebook's Config cell
COLLAPSE_F1 = 0.1
CHECKPOINT_URL = re.compile(r"https://huggingface\.co/[^/\s]+/[^/\s]+/tree/[0-9a-f]{40}")
RETIRED_SCORES = ("0.8330", "0.8343", "0.0807")  # pre-restart runs, must not be reported


def check(cond: bool, msg: str) -> None:
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        raise SystemExit(f"Gate failed: {msg}")


def parse_notes(notes: str) -> dict[str, str]:
    return dict(part.split("=", 1) for part in notes.split("; ") if "=" in part)


def read_text(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def notebook_stdout(nb: dict) -> list[str]:
    """Printed lines from every code cell, in cell order."""
    lines = []
    for cell in nb["cells"]:
        text = "".join("".join(out.get("text", [])) for out in cell.get("outputs", [])
                       if out.get("output_type") == "stream" and out.get("name") == "stdout")
        lines.extend(text.splitlines())
    return lines


def choose_subset_size(curve_rows: list[dict[str, str]]) -> int:
    """Smallest size with no collapsed seed whose mean val F1 is within tolerance of the best."""
    val_by_size = defaultdict(list)
    for r in curve_rows:
        val_by_size[int(r["subset_size"])].append(float(r["parsed_notes"]["val_F1"]))
    means = {n: mean(f1s) for n, f1s in val_by_size.items()}
    best = max(means.values())
    for n in sorted(val_by_size):
        if min(val_by_size[n]) >= COLLAPSE_F1 and means[n] >= best - SIZE_TOLERANCE:
            return n
    raise SystemExit("Gate failed: no subset size satisfies the selection rule")


def main() -> None:
    print("1. Results sheet")
    check(os.path.exists(RESULTS_PATH), f"{os.path.relpath(RESULTS_PATH, REPO_ROOT)} exists")
    with open(RESULTS_PATH, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = list(reader)
    check(header == COLUMNS, "header matches the fixed columns")

    baseline_runs = [r for r in rows if r["model"] == BASELINE_MODEL and r["train_lang"] == LANG and r["test_lang"] == LANG
                     and "DEBUG" not in r["notes"]]
    check(len(baseline_runs) >= 1, f"{BASELINE_MODEL} {LANG}->{LANG} baseline row present (Phase 2)")
    baseline_f1 = max(float(r["F1_overall"]) for r in baseline_runs)
    print(f"  baseline (BiLSTM-CRF) Hindi gold F1 = {baseline_f1:.4f}")

    phase_rows = [r for r in rows if r["model"] in PHASE_MODELS and r["train_lang"] == LANG and r["test_lang"] == LANG
                  and "phase=4" not in r["notes"]]
    check(len(phase_rows) >= 1, "Phase 3 rows logged")
    check(not any("DEBUG" in r["notes"] for r in phase_rows), "no DEBUG rows committed")

    print("\n2. Run setup and checkpoints")
    for r in phase_rows:
        r["parsed_notes"] = notes = parse_notes(r["notes"])
        name = f"{r['model']} n={r['subset_size']} e={r['epochs']} seed={notes.get('seed')} lr={notes.get('lr')}"
        missing = [k for k in REQUIRED_NOTES if k not in notes]
        check(not missing, f"{name}: notes record the run setup (missing: {missing or 'none'})")
        check(notes["gpus"] == "1" and notes["effective_bs"] == "32", f"{name}: single GPU, effective batch 32")
        check(int(notes["warmup_steps"]) > 0, f"{name}: warmup applied")
        for c in SCORE_COLS:
            check(0.0 <= float(r[c]) <= 1.0, f"{name}: {c} is a valid entity F1")
        check(bool(CHECKPOINT_URL.fullmatch(r["checkpoint_path"])), f"{name}: checkpoint_path is a pinned Hugging Face link")
    commits = {r["parsed_notes"]["git"] for r in phase_rows}
    check(len(commits) == 1, f"all Phase 3 rows come from one code commit ({sorted(commits)})")

    print("\n3. One clean notebook execution")
    check(os.path.exists(NOTEBOOK), f"{os.path.relpath(NOTEBOOK, REPO_ROOT)} exists")
    nb = json.loads(read_text(NOTEBOOK))
    counts = [c.get("execution_count") for c in nb["cells"] if c["cell_type"] == "code"]
    check(counts == list(range(1, len(counts) + 1)), "every code cell ran once, in order, from a fresh kernel")
    stdout = notebook_stdout(nb)
    printed = [tuple(row) for row in csv.reader(l for l in stdout if l.startswith(tuple(m + "," for m in PHASE_MODELS)))]
    logged = [tuple(r[c] for c in COLUMNS) for r in phase_rows]
    check(Counter(printed) == Counter(logged),
          f"results rows exactly match the notebook's printed rows ({len(logged)} logged, {len(printed)} printed)")

    print("\n4. MuRIL scaling curve and size choice")
    muril = [r for r in phase_rows if r["model"] == "MuRIL"]
    curve = [r for r in muril if r["epochs"] == CURVE_EPOCHS and r["parsed_notes"]["lr"] == CURVE_LR]
    sizes = sorted({int(r["subset_size"]) for r in curve})
    check(len(sizes) >= MIN_SCALING_POINTS, f"at least {MIN_SCALING_POINTS} distinct subset sizes on the curve ({sizes})")
    chosen = choose_subset_size(curve)
    stated = [int(m.group(1)) for l in stdout if (m := re.fullmatch(r"working subset size: (\d+)", l))]
    check(stated == [chosen], f"notebook's chosen size {stated} matches the validation-F1 rule ({chosen})")
    chosen_rows = [r for r in curve if int(r["subset_size"]) == chosen]
    seeds = {r["parsed_notes"]["seed"] for r in chosen_rows}
    check(REQUIRED_SEEDS <= seeds, f"chosen size ran on seeds {sorted(REQUIRED_SEEDS)} (found {sorted(seeds)})")
    handoff = [r for r in chosen_rows if r["parsed_notes"]["seed"] == HANDOFF_SEED]
    check(len(handoff) == 1, f"exactly one handoff row at n={chosen}, seed {HANDOFF_SEED}")
    handoff = handoff[0]
    handoff_f1 = float(handoff["F1_overall"])
    margin = handoff_f1 - baseline_f1
    check(margin >= MIN_MARGIN, f"handoff MuRIL F1 {handoff_f1:.4f} beats baseline {baseline_f1:.4f} by >= {MIN_MARGIN} (margin {margin:.4f})")
    verified = [m for l in stdout if (m := re.match(r"handoff verified: (\S+)@([0-9a-f]{40}) ", l))]
    check(len(verified) == 1 and handoff["checkpoint_path"] == f"https://huggingface.co/{verified[0][1]}/tree/{verified[0][2]}",
          "handoff checkpoint was reloaded from its pinned revision and re-scored")

    print("\n5. IndicBERT comparison")
    indicbert = [r for r in phase_rows if r["model"] == "IndicBERT" and int(r["subset_size"]) == chosen]
    check(len(indicbert) == 1, f"one IndicBERT row at the chosen size (n={chosen})")

    print("\n6. Phase 3 docs")
    check(os.path.exists(PHASE_DOC), f"{os.path.relpath(PHASE_DOC, REPO_ROOT)} exists")
    doc = read_text(PHASE_DOC)
    roadmap = read_text(ROADMAP)
    roadmap_phase = roadmap[roadmap.index("## Phase 3"):roadmap.index("## Phase 4")]
    check(f"{handoff_f1:.4f}" in doc, f"README reports the handoff F1 {handoff_f1:.4f}")
    doc_size = re.search(r"chosen working subset size:\**\s*([\d,]+)", doc, re.IGNORECASE)
    check(bool(doc_size) and int(doc_size.group(1).replace(",", "")) == chosen, f"README states the chosen size {chosen:,}")
    check(f"{handoff_f1:.4f}" in roadmap_phase and f"n={chosen:,}" in roadmap_phase, "roadmap reports the same F1 and size")
    decision = re.search(r"phase 3 working subset size[^\n]*?(\d[\d,]{3,})", read_text(DECISIONS), re.IGNORECASE)
    check(bool(decision) and int(decision.group(1).replace(",", "")) == chosen, "decisions log records the chosen size")
    stale = [s for s in RETIRED_SCORES if s in doc or s in roadmap_phase]
    check(not stale, f"docs do not report retired pre-restart scores ({stale or 'none'})")

    print(f"\n  MuRIL Hindi gold F1 (handoff, n={chosen}): {handoff_f1:.4f}")
    print(f"  Baseline (BiLSTM-CRF): {baseline_f1:.4f}")
    print("\nPhase 3 gate: PASSED")


if __name__ == "__main__":
    main()
