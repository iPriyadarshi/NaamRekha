"""Phase 1 gate check. Uses only the shared code in src/.

Gate (docs/01-roadmap.md): load Hindi, print the IOB2 tag distribution, and
score a dummy all-O prediction with seqeval. Also checks that Marathi and
Bengali load with the same format, since Phase 4 depends on them.

Run from the repo root:  uv run python scripts/phase1_gate.py
"""

from __future__ import annotations

import csv
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import (  # noqa: E402
    COLUMNS,
    LABELS,
    SPLITS,
    evaluate,
    id2label,
    label2id,
    load_split,
    log_result,
    tag_distribution,
)

EXPECTED_LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
HI_TEST_ROWS = 864  # 867 rows in the file minus 3 empty ones (docs/phase1)


def check(cond: bool, msg: str) -> None:
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        raise SystemExit(f"Gate failed: {msg}")


def check_rows(lang: str, split: str, rows) -> None:
    check(len(rows) > 0, f"{lang}/{split}: {len(rows):,} sentences loaded")
    check(all(w for w, _ in rows), f"{lang}/{split}: no empty sentences")
    check(all(len(w) == len(t) for w, t in rows), f"{lang}/{split}: words and tags align")
    unknown = set(tag_distribution(rows)) - set(LABELS)
    check(not unknown, f"{lang}/{split}: only known tags (extra: {sorted(unknown)})")


def print_distribution(rows) -> None:
    counts = tag_distribution(rows)
    total = sum(counts.values())
    for label in LABELS:
        n = counts.get(label, 0)
        print(f"    {label:<6} {n:>8,}  ({n / total:6.2%})")


def main() -> None:
    print("1. Label map")
    check(LABELS == EXPECTED_LABELS, "label order matches the Phase 1 contract")
    check(all(id2label[label2id[x]] == x for x in LABELS), "label2id / id2label round-trip")

    print("\n2. Hindi test split")
    hi_test = load_split("hi", "test")
    check_rows("hi", "test", hi_test)
    check(len(hi_test) == HI_TEST_ROWS, f"hi/test has {HI_TEST_ROWS} sentences")
    print("  Tag distribution (hi/test):")
    print_distribution(hi_test)

    print("\n3. Evaluator")
    gold = [tags for _, tags in hi_test]
    all_o = [["O"] * len(tags) for tags in gold]
    scores = evaluate(gold, all_o)
    print(f"  all-O scores: {scores}")
    check(all(v == 0.0 for v in scores.values()), "all-O prediction scores 0.0 everywhere")
    perfect = evaluate(gold, gold)
    print(f"  gold-vs-gold scores: {perfect}")
    check(all(v == 1.0 for v in perfect.values()), "gold vs itself scores 1.0 everywhere")

    print("\n4. Results logger (temp file, real results.csv untouched)")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "results.csv")
        for _ in range(2):
            log_result(model="dummy-all-O", train_lang="none", test_lang="hi",
                       subset_size=0, epochs=0, scores=scores, path=path)
        with open(path, encoding="utf-8", newline="") as f:
            rows = list(csv.reader(f))
        check(rows[0] == COLUMNS, "header matches the fixed columns")
        check(len(rows) == 3, "header written once, one row per call")

    print("\n5. All splits for hi / mr / bn")
    for lang in ("hi", "mr", "bn"):
        for split in SPLITS:
            if lang == "hi" and split == "test":
                continue  # already checked above
            check_rows(lang, split, load_split(lang, split))
        if lang != "hi":
            print(f"  Tag distribution ({lang}/test):")
            print_distribution(load_split(lang, "test"))

    print("\nPhase 1 gate: PASSED")


if __name__ == "__main__":
    main()
