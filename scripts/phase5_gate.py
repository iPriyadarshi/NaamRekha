"""Phase 5 gate check. Uses only the shared code in src/ and committed files.

Gate (docs/01-roadmap.md): LLM entity-F1 (zero-shot and few-shot) is in the
results sheet with the model name and sample size recorded, and one page of
concrete error examples is written.

Beyond that, this script checks that the two rows come from one clean
top-to-bottom notebook run of one commit on all 864 gold sentences, that the
model revision is pinned, that malformed output was counted (not dropped),
and that the F1 in the sheet can be reproduced by re-scoring the saved
per-sentence predictions with the shared evaluator. It does not run any model.

Run from the repo root:  uv run python scripts/phase5_gate.py
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
from collections import Counter

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from src import COLUMNS, RESULTS_PATH, evaluate  # noqa: E402

LANG = "hi"
FULL_TEST = 864
NOTEBOOK = os.path.join(REPO_ROOT, "notebooks", "phase5-llm-prompting.ipynb")
PHASE_DOC = os.path.join(REPO_ROOT, "docs", "phase5", "README.md")
ERROR_DOC = os.path.join(REPO_ROOT, "docs", "phase5", "error_analysis.md")
ROADMAP = os.path.join(REPO_ROOT, "docs", "01-roadmap.md")
PREDICTIONS = os.path.join(REPO_ROOT, "results", "phase5", "final_predictions.jsonl")
SCORE_COLS = ["F1_PER", "F1_ORG", "F1_LOC", "F1_overall"]
MODES = ("zero-shot", "few-shot")
REQUIRED_NOTES = ("phase", "mode", "k_shot", "test_n", "seed", "model", "revision", "prompt", "parser", "config",
                  "malformed", "malformed_rate", "git", "torch", "transformers")
CHECKPOINT_URL = re.compile(r"https://huggingface\.co/([^/\s]+/[^/\s]+)/tree/([0-9a-f]{40})")
ERROR_KINDS = ("missed", "boundary", "spurious", "type confusion", "format")
RETIRED_SCORES = ("0.5335", "0.5741")   # the earlier 200-sentence gold-test draft, not a result of this phase


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
    lines = []
    for cell in nb["cells"]:
        text = "".join("".join(out.get("text", [])) for out in cell.get("outputs", [])
                       if out.get("output_type") == "stream" and out.get("name") == "stdout")
        lines.extend(text.splitlines())
    return lines


def main() -> None:
    print("1. Parser tests (no GPU)")
    import test_phase5_parser as tp  # noqa: E402

    tests = [f for n, f in sorted(vars(tp).items()) if n.startswith("test_") and callable(f)]
    for t in tests:
        t()
    check(len(tests) >= 20, f"all {len(tests)} parser tests pass")

    print("\n2. Results sheet")
    check(os.path.exists(RESULTS_PATH), f"{os.path.relpath(RESULTS_PATH, REPO_ROOT)} exists")
    with open(RESULTS_PATH, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = list(reader)
    check(header == COLUMNS, "header matches the fixed columns")
    llm = [r for r in rows if r["model"].startswith("LLM-") and r["test_lang"] == LANG]
    check(not any("DEBUG" in r["notes"] for r in llm), "no DEBUG rows committed")
    by_mode = {}
    for mode in MODES:
        found = [r for r in llm if r["model"].startswith(f"LLM-{mode} (")]
        check(len(found) == 1, f"exactly one {mode} row logged ({len(found)} found)")
        by_mode[mode] = found[0]
    check(len(llm) == 2, "no other LLM rows in the sheet (earlier drafts removed)")

    print("\n3. Rows follow docs/03-conventions.md and the Phase 5 rules")
    for mode, r in by_mode.items():
        r["parsed"] = n = parse_notes(r["notes"])
        missing = [k for k in REQUIRED_NOTES if k not in n]
        check(not missing, f"{mode}: notes record the setup (missing: {missing or 'none'})")
        check(n["phase"] == "5" and n["mode"] == mode, f"{mode}: phase=5 and mode matches the model name")
        for c in SCORE_COLS:
            check(0.0 < float(r[c]) <= 1.0, f"{mode}: {c} = {float(r[c]):.4f} is a valid entity F1")
        check(r["train_lang"] == "none" and r["subset_size"] == "0" and r["epochs"] == "0",
              f"{mode}: no training (train_lang=none, subset_size=0, epochs=0)")
        used, total = (int(x) for x in n["test_n"].split("/"))
        check(used == total == FULL_TEST, f"{mode}: scored on all {FULL_TEST} gold sentences (test_n={n['test_n']})")
        m = CHECKPOINT_URL.fullmatch(r["checkpoint_path"])
        check(bool(m), f"{mode}: checkpoint_path is a pinned Hugging Face URL")
        check(m.group(1) == n["model"] and m.group(2) == n["revision"], f"{mode}: URL matches model and revision in notes")
        check(n["malformed"].isdigit() and abs(int(n["malformed"]) / FULL_TEST - float(n["malformed_rate"])) < 1e-4,
              f"{mode}: malformed output counted ({n['malformed']} of {FULL_TEST}, rate {n['malformed_rate']})")
    z, f_ = by_mode["zero-shot"]["parsed"], by_mode["few-shot"]["parsed"]
    check(z["k_shot"] == "0" and int(f_["k_shot"]) >= 1, f"few-shot uses {f_['k_shot']} examples, zero-shot uses 0")
    for k in ("model", "revision", "git", "prompt", "parser", "config", "seed"):
        check(z[k] == f_[k], f"both rows share {k} ({z[k][:12]})")

    print("\n4. One clean notebook execution")
    check(os.path.exists(NOTEBOOK), f"{os.path.relpath(NOTEBOOK, REPO_ROOT)} exists")
    nb = json.loads(read_text(NOTEBOOK))
    counts = [c.get("execution_count") for c in nb["cells"] if c["cell_type"] == "code"]
    check(counts == list(range(1, len(counts) + 1)), "every code cell ran once, in order, from a fresh kernel")
    errors = [o for c in nb["cells"] for o in c.get("outputs", []) if o.get("output_type") == "error"]
    check(not errors, "no cell raised an error")
    stdout = notebook_stdout(nb)
    check(any(l.startswith("stage: final | debug: False") for l in stdout), "the executed run was STAGE=final, DEBUG=False")
    tests_line = [m for l in stdout if (m := re.search(r"(\d+)/(\d+) parser tests passed", l))]
    check(len(tests_line) == 1 and tests_line[0][1] == tests_line[0][2], "parser tests passed at the start of the run")
    commit = [m.group(1) for l in stdout if (m := re.search(r"\| commit: (\w+)", l))]
    check(commit == [z["git"]], f"notebook ran commit {commit}, rows record {z['git']}")
    printed = [tuple(r) for r in csv.reader(l for l in stdout if l.startswith("LLM-"))]
    logged = [tuple(r[c] for c in COLUMNS) for r in by_mode.values()]
    check(Counter(printed) == Counter(logged), "results rows exactly match the notebook's printed rows")
    check(any("reproduces the Phase 3 handoff score" in l for l in stdout),
          "MuRIL comparison reproduced Phase 3's 0.8274 on the same sentences")

    print("\n5. Saved predictions reproduce the logged F1")
    check(os.path.exists(PREDICTIONS), f"{os.path.relpath(PREDICTIONS, REPO_ROOT)} exists")
    recs: dict[str, list[dict]] = {}
    with open(PREDICTIONS, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            recs.setdefault(rec["system"], []).append(rec)
    for mode, r in by_mode.items():
        system = next((s for s in recs if s.startswith(f"LLM {mode} [")), None)
        check(system is not None, f"{mode}: predictions saved")
        rs = sorted(recs[system], key=lambda x: x["idx"])
        check(len(rs) == FULL_TEST, f"{mode}: {len(rs)} sentences saved")
        check(all(len(x["pred"]) == len(x["gold"]) == len(x["words"]) for x in rs), f"{mode}: one tag per word everywhere")
        scores = evaluate([x["gold"] for x in rs], [x["pred"] for x in rs])
        for c in SCORE_COLS:
            check(abs(scores[c] - float(r[c])) < 1e-9, f"{mode}: re-scored {c} = {scores[c]:.4f} matches the sheet")
        n_bad = sum(bool(x["malformed"]) for x in rs)
        check(n_bad == int(r["parsed"]["malformed"]), f"{mode}: {n_bad} malformed outputs in the saved raw outputs match the sheet")
        check(all(x["pred"] == ["O"] * len(x["pred"]) for x in rs if x["malformed"]),
              f"{mode}: every malformed sentence was scored as all O (not dropped)")
    check(any(s.startswith("MuRIL") for s in recs), "MuRIL predictions saved for the error comparison")

    print("\n6. Docs")
    for path in (PHASE_DOC, ERROR_DOC):
        check(os.path.exists(path), f"{os.path.relpath(path, REPO_ROOT)} exists")
    doc, err_doc = read_text(PHASE_DOC), read_text(ERROR_DOC)
    for mode, r in by_mode.items():
        check(f"{float(r['F1_overall']):.4f}" in doc, f"README reports the {mode} F1 {float(r['F1_overall']):.4f}")
        check(f"{r['parsed']['malformed']} malformed" in doc.lower() or f"malformed: {r['parsed']['malformed']}" in doc.lower(),
              f"README reports the {mode} malformed count ({r['parsed']['malformed']})")
    check(z["model"] in doc and z["revision"] in doc, "README names the model and its pinned revision")
    check(str(FULL_TEST) in doc, "README states the test size")
    check(all(k in err_doc.lower() for k in ERROR_KINDS), f"error analysis covers: {', '.join(ERROR_KINDS)}")
    check(len(re.findall(r"[ऀ-ॿ]+", err_doc)) >= 30, "error analysis has concrete Hindi examples")
    check(len(err_doc.splitlines()) <= 120, "error analysis stays about one page")
    stale = [s for s in RETIRED_SCORES if s in doc or s in err_doc]
    check(not stale, f"docs do not report the retired 200-sentence draft scores ({stale or 'none'})")
    roadmap = read_text(ROADMAP)
    section = roadmap[roadmap.index("## Phase 5"):roadmap.index("## Phase 6")]
    check(all(f"{float(r['F1_overall']):.4f}" in section for r in by_mode.values()), "roadmap Phase 5 reports both F1 scores")

    print("\n  LLM Hindi gold F1 (" + z["model"] + ", all 864 sentences):")
    for mode, r in by_mode.items():
        print(f"    {mode:9s} overall {float(r['F1_overall']):.4f} | PER {float(r['F1_PER']):.4f} | "
              f"ORG {float(r['F1_ORG']):.4f} | LOC {float(r['F1_LOC']):.4f} | malformed {r['parsed']['malformed']}")
    print("\nPhase 5 gate: PASSED")


if __name__ == "__main__":
    main()
