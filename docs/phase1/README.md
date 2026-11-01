# Phase 1: Data Inspection

Everything in this folder is specific to Phase 1. The job of this phase was to look at the real Naamapadam data and turn what we saw into a fixed data contract that `src/` and every later notebook depend on. Nothing here was assumed; all of it was read directly from the distributed dataset files in [phase1-infra.ipynb](../../notebooks/phase1-infra.ipynb).

## What we inspected

Source: [ai4bharat/naamapadam](https://huggingface.co/datasets/ai4bharat/naamapadam). The dataset ships as per-language zip archives (`data/{lang}_IndicNER_v1.0.zip`), each holding `{lang}_train.json`, `{lang}_val.json`, `{lang}_test.json`. Every file is JSON-lines: one object per line, `{"words": [...tokens...], "ner": [...tag strings...]}`. The `ner` field holds tag strings such as `B-PER`, not integer ids.

Phase 1 inspected the Hindi splits in full.

## Findings

- **Tag set:** exactly `O`, `B-PER`, `I-PER`, `B-ORG`, `I-ORG`, `B-LOC`, `I-LOC`. No other labels appear in any split. Scheme is IOB2.
- **Alignment:** `words` and `ner` are always equal length. Zero mismatches across all splits.
- **Row counts (Hindi):** train 985,787 / val 13,460 / test 867.
- **Empty rows:** 3 zero-length rows in test, 0 in train and val.
- **Sentence length (train, in words):** min 1, median 20, p95 48, p99 67, max 100. The dataset is pre-capped at 100 words.
- **Label noise:** orphan `I-` tags (a continuation tag with no matching start of the same type) appear 5,513 times in train, 74 in val, 7 in test. This is roughly 0.5% at the token level and comes from the dataset's automatic label projection.

## Contract decisions

These are binding for all later phases and are also recorded in [../06-decisions.md](../06-decisions.md).

1. **Label set and order are fixed** as the seven tags above, in this order:
   `["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]`.
   The shared loader defines `label2id` / `id2label` once from this list and everyone imports it. See [../03-conventions.md](../03-conventions.md).
2. **Orphan `I-` tags are left as-is.** We do not repair them. Repairing would move our numbers away from the published Naamapadam benchmark, and `seqeval` scores them deterministically (it treats an orphan `I-` as the start of a fresh entity). Recorded as a known quirk.
3. **`max_length` = 128 subword tokens.** Word length tops out at 100, which stays under 128 even after subword expansion, so no real sentence is truncated.
4. **The loader drops zero-length sentences on every split.** This removes the 3 empty test rows that would otherwise break evaluation.

## What this unblocks

The shared loader and evaluation harness in `src/` can now be written against a known contract:

- `load_split(lang, split)` returns a list of `(words, tags)` pairs, tags as strings, empty rows dropped.
- A single `label2id` / `id2label` built from the fixed label list.
- A `seqeval` entity-level micro-F1 scorer, reported per entity type and overall, per [../03-conventions.md](../03-conventions.md).
