"""Phase 6 gate check. Uses only the shared code in src/ and committed files.

Gate (docs/01-roadmap.md): the report is done and every number in it is
traceable to a logged run.

This script recomputes each number the report states from its source and
checks that the report says the same thing:
  * every F1 in the tables from results/results.csv
  * the transfer matrix, its spreads and the paired gaps (Phase 4 rows)
  * the LLM rows and their malformed counts
  * error counts from results/error_analysis/ and results/phase5/final_errors.csv
  * the "right in all four systems" counts, rebuilt from the saved predictions
    and the real Hindi gold test split
  * tokenizer statistics from results/error_analysis/token_stats.csv
It also checks that no placeholder is left, that every link and figure in the
report exists, and that the figure script can read its data.

Run from the repo root:  uv run python scripts/phase6_gate.py
"""

from __future__ import annotations

import csv
import gzip
import json
import os
import re
import sys
from collections import Counter, defaultdict
from statistics import mean, stdev

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from seqeval.metrics.sequence_labeling import get_entities  # noqa: E402

from phase5_parser import format_entities, parse_llm_output, tags_to_entities  # noqa: E402
from src import RESULTS_PATH, evaluate, id2label, load_split  # noqa: E402

REPORT = os.path.join(REPO_ROOT, "docs", "phase6", "report.md")
ERR_DIR = os.path.join(REPO_ROOT, "results", "error_analysis")
FINAL_ERRORS = os.path.join(REPO_ROOT, "results", "phase5", "final_errors.csv")
FIGURES = ("phase3_scaling_curve.png", "phase4_transfer_matrix.png", "phase6_hindi_comparison.png", "phase6_error_kinds.png")
PLACEHOLDERS = ("PENDING", "TODO", "TBD", "XXX")
FOUR = ("BiLSTM-CRF", "MuRIL-hi-s42", "IndicBERT", "LLM-few")


def check(cond: bool, msg: str) -> None:
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        raise SystemExit(f"Gate failed: {msg}")


def read_csv(path: str) -> list[dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def notes_of(row: dict) -> dict[str, str]:
    return dict(p.split("=", 1) for p in row["notes"].split("; ") if "=" in p)


def f4(x: float) -> str:
    return f"{x:.4f}"


def main() -> None:
    check(os.path.exists(REPORT), f"{os.path.relpath(REPORT, REPO_ROOT)} exists")
    with open(REPORT, encoding="utf-8") as f:
        raw = f.read()
    text = raw.replace("*", "").replace("`", "").replace("−", "-")   # compare without bold, code marks or the minus sign

    def says(s: str, what: str) -> None:
        check(s in text, f"report states {what}: {s}")

    print("1. Report hygiene")
    check(not any(p in raw for p in PLACEHOLDERS), f"no placeholder left ({', '.join(PLACEHOLDERS)})")
    check("## Appendix A" in raw, "Appendix A (where each number comes from) is present")
    links = re.findall(r"\]\((?!https?:)([^)#\s]+)", raw)
    missing = [l for l in links if not os.path.exists(os.path.normpath(os.path.join(os.path.dirname(REPORT), l)))]
    check(not missing, f"all {len(links)} relative links resolve ({missing or 'none broken'})")
    for fig in FIGURES:
        check(fig in raw and os.path.exists(os.path.join(REPO_ROOT, "results", fig)), f"figure {fig} exists and is in the report")
    import phase6_figures as pf

    pf.load_scores(), pf.load_errors()
    check(True, "figure script reads its data from the logged results")

    rows = read_csv(RESULTS_PATH)
    phase = lambda r: notes_of(r).get("phase")
    early = [r for r in rows if phase(r) is None]
    num = lambda r, k: float(r[k])

    print("\n2. Hindi stage scores (results.csv)")
    bil = [r for r in rows if r["model"] == "BiLSTM-CRF"]
    ind = [r for r in rows if r["model"] == "IndicBERT"]
    hand = [r for r in early if r["model"] == "MuRIL" and r["train_lang"] == r["test_lang"] == "hi" and r["subset_size"] == "300000"
            and notes_of(r)["seed"] == "42"]
    check(len(bil) == len(ind) == len(hand) == 1, "one BiLSTM-CRF, one IndicBERT and one MuRIL handoff row")
    for name, r in (("BiLSTM-CRF", bil[0]), ("MuRIL handoff", hand[0]), ("IndicBERT", ind[0])):
        for k in ("F1_PER", "F1_ORG", "F1_LOC", "F1_overall"):
            says(f4(num(r, k)), f"{name} {k}")
    seeds3 = [r for r in early if r["model"] == "MuRIL" and r["train_lang"] == r["test_lang"] == "hi" and r["subset_size"] == "300000"
              and r["epochs"] == "3"]
    check(sorted(notes_of(r)["seed"] for r in seeds3) == ["42", "43", "44"], "MuRIL 300k has seeds 42, 43, 44 in Phase 3")
    says(f4(mean(num(r, "F1_overall") for r in seeds3)), "Phase 3 three-seed MuRIL mean")
    says(f4(min(num(r, "F1_overall") for r in seeds3)), "lowest MuRIL seed")

    print("\n3. Cross-lingual transfer (Phase 4 rows)")
    p4 = [r for r in rows if phase(r) == "4"]
    check(len(p4) == 15, f"15 Phase 4 rows ({len(p4)} found)")
    groups: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in p4:
        n = notes_of(r)
        key = {("hi", "hi"): "hi-hi", ("hi", "mr"): "hi-mr", ("hi", "bn"): "hi-bn", ("mr", "mr"): "mr-mr", ("bn", "bn"): "bn-bn"}[
            (r["train_lang"], r["test_lang"])]
        groups[key][n["seed"]] = r
    check(all(sorted(g) == ["42", "43", "44"] for g in groups.values()) and len(groups) == 5, "five matrix cells, three seeds each")
    for key in groups:
        vals = [num(groups[key][s], "F1_overall") for s in ("42", "43", "44")]
        says(f"{f4(mean(vals))} ± {f4(stdev(vals))}", f"mean ± SD of {key}")
    for tgt, name in (("mr", "Marathi"), ("bn", "Bengali")):
        gaps = {s: num(groups[f"{tgt}-{tgt}"][s], "F1_overall") - num(groups[f"hi-{tgt}"][s], "F1_overall") for s in ("42", "43", "44")}
        g = list(gaps.values())
        says(f"{mean(g):+.4f} (SD {stdev(g):.4f})", f"{name} transfer gap")
        says(" / ".join(f"{x:+.4f}" for x in g), f"{name} per-seed gaps")
        org = mean(num(groups[f"{tgt}-{tgt}"][s], "F1_ORG") - num(groups[f"hi-{tgt}"][s], "F1_ORG") for s in ("42", "43", "44"))
        says(f"{name} {org:+.4f}", f"{name} ORG gap")
    for k in ("F1_PER", "F1_ORG", "F1_LOC"):
        says(f4(mean(num(groups["hi-hi"][s], k) for s in ("42", "43", "44"))), f"Phase 4 hi to hi mean {k}")

    print("\n4. LLM rows (results.csv)")
    llm = {m: next(r for r in rows if r["model"].startswith(f"LLM-{m} (")) for m in ("zero-shot", "few-shot")}
    for m, r in llm.items():
        mal = notes_of(r)["malformed"]
        label = "LLM few-shot (k=8)" if m == "few-shot" else "LLM zero-shot"      # as labelled in the report table
        says(f"{label} | {f4(num(r, 'F1_PER'))} | {f4(num(r, 'F1_ORG'))} | {f4(num(r, 'F1_LOC'))} | {f4(num(r, 'F1_overall'))} | {mal} |",
             f"{m} row with its malformed count")
    z, fs, mu = llm["zero-shot"], llm["few-shot"], hand[0]
    says(f"{num(fs, 'F1_overall') - num(z, 'F1_overall'):.4f} F1", "the few-shot gain")
    d = {k: num(fs, k) - num(mu, k) for k in ("F1_PER", "F1_ORG", "F1_LOC")}
    says(f"PER ({d['F1_PER']:+.3f})", "few-shot minus MuRIL on PER")
    says(f"ORG {d['F1_ORG']:+.3f}", "few-shot minus MuRIL on ORG")
    says(f"LOC {d['F1_LOC']:+.3f}", "few-shot minus MuRIL on LOC")

    print("\n5. Error counts")
    ec = {(r["lang"], r["system"]): {k: int(r[k]) for k in ("boundary", "missed", "spurious", "type_confusion", "total")}
          for r in read_csv(os.path.join(ERR_DIR, "error_counts.csv"))}
    says(f"MuRIL makes the fewest errors ({ec[('hi', 'MuRIL-hi-s42')]['total']:,}), then BiLSTM-CRF ({ec[('hi', 'BiLSTM-CRF')]['total']:,}), "
         f"IndicBERT ({ec[('hi', 'IndicBERT')]['total']:,}), LLM few-shot ({ec[('hi', 'LLM-few')]['total']:,}), "
         f"and LLM zero-shot ({ec[('hi', 'LLM-zero')]['total']:,})", "the five Hindi error totals")
    m, b = ec[("hi", "MuRIL-hi-s42")], ec[("hi", "BiLSTM-CRF")]
    says(f"({m['missed']} vs {b['missed']} missed)", "MuRIL vs BiLSTM-CRF misses")
    says(f"({m['spurious']} vs {b['spurious']})", "MuRIL vs BiLSTM-CRF spurious spans")
    says(f"spurious spans ({m['spurious']} of {m['total']}) and boundaries ({m['boundary']})", "MuRIL's main error kinds")
    fw, zs = ec[("hi", "LLM-few")], ec[("hi", "LLM-zero")]
    says(f"cut spurious spans and misses by {zs['spurious'] - fw['spurious']} each and add {fw['boundary'] - zs['boundary']} boundary errors",
         "few-shot against zero-shot")
    check(zs["spurious"] - fw["spurious"] == zs["missed"] - fw["missed"], "spurious and missed drop by the same amount, as the report says")
    fe = read_csv(FINAL_ERRORS)
    typ = lambda r: r["gold_type"] or r["pred_type"]
    few = [r for r in fe if r["system"].startswith("LLM few")]
    mur = [r for r in fe if r["system"].startswith("MuRIL")]
    bd = [r for r in few if r["kind"] == "boundary"]
    longer = sum(len(r["pred"].split()) > len(r["gold"].split()) for r in bd)
    says(f"{longer} of {len(bd)} longer than gold", "over-long boundary errors")
    lo = lambda rs: sum(r["kind"] == "type_confusion" and r["gold_type"] == "LOC" and r["pred_type"] == "ORG" for r in rs)
    says(f"LOC as ORG: {lo(few)} LLM, {lo(mur)} MuRIL", "LOC read as ORG")
    morg = [r for r in mur if typ(r) == "ORG"]
    says(f"ORG alone has {sum(r['kind'] == 'spurious' for r in morg)} spurious spans and {sum(r['kind'] == 'boundary' for r in morg)} boundary errors",
         "MuRIL's ORG errors")
    for tgt in ("mr", "bn"):
        pair = lambda kind: [ec[(tgt, f"MuRIL-{tgt}-s{s}")][kind] - ec[(tgt, f"MuRIL-hi-s{s}")][kind] for s in (42, 43, 44)]
        miss, spur = pair("missed"), pair("spurious")
        if tgt == "mr":
            check(all(x > 0 for x in miss) and all(x < 0 for x in spur), "Marathi: more misses and fewer spurious spans on every seed")
            says(f"Marathi in-language models miss about {round(mean(miss))} more entities than Hindi models and make about "
                 f"{round(abs(mean(spur)))} fewer spurious predictions", "the Marathi error shift")
        else:
            check(all(x < 0 for x in miss), "Bengali: fewer misses on every seed")
            says(f"Bengali in-language models miss about {round(abs(mean(miss)))} fewer entities than Hindi models", "the Bengali error shift")

    print("\n6. Cross-stage counts rebuilt from saved predictions and the real gold split")
    gold = [t for _, t in load_split("hi", "test")]
    preds: dict[str, dict[int, list[str]]] = defaultdict(dict)
    with gzip.open(os.path.join(ERR_DIR, "predictions.jsonl.gz"), "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["lang"] == "hi" and r["system"] in FOUR:
                preds[r["system"]][r["idx"]] = [id2label[int(c)] for c in r["pred"]]   # saved as one label id per token
    check(all(len(preds[s]) == len(gold) for s in FOUR), "predictions saved for all four systems on all 864 sentences")
    check(all(len(preds[s][i]) == len(g) for s in FOUR for i, g in enumerate(gold)), "one tag per word in every saved prediction")
    all_right = all_wrong = 0
    wrong_by_type: Counter = Counter()
    total_by_type: Counter = Counter()
    for i, g in enumerate(gold):
        for t, s, e in get_entities(g):
            hits = [(t, s, e) in {(tt, ss, ee) for tt, ss, ee in get_entities(preds[sys_][i])} for sys_ in FOUR]
            total_by_type[t] += 1
            all_right += all(hits)
            if not any(hits):
                all_wrong += 1
                wrong_by_type[t] += 1
    says(f"{all_right:,} are right in all four", "entities right in all four systems")
    says(f"{all_wrong} are wrong in all four, including {wrong_by_type['ORG']} ORG", "entities wrong in all four systems")
    says(f"Of {sum(total_by_type.values()):,} gold entities", "the gold entity total")

    print("\n7. Tokenizer statistics")
    ts = {r["lang"]: r for r in read_csv(os.path.join(ERR_DIR, "token_stats.csv"))}
    says(f"{float(ts['bn']['pct_words_with_unk']):.3f}% of Bengali words", "Bengali [UNK] share")
    says(f"{float(ts['bn']['mean_subwords']):.3f} vs {float(ts['mr']['mean_subwords']):.3f}", "subwords per word, Bengali vs Marathi")

    print("\n8. Parser ceiling")
    rt = [parse_llm_output(format_entities(tags_to_entities(w, t)), w).tags for w, t in load_split("hi", "test")]
    says(f4(evaluate(gold, rt)["F1_overall"]), "the parser's round-trip ceiling on gold")

    print("\nPhase 6 gate: PASSED")


if __name__ == "__main__":
    main()
