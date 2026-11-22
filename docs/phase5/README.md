# Phase 5: LLM Prompting Baseline

> **Status: draft. No LLM results yet.** The sections marked **PENDING** are filled in after the validation and final Kaggle runs. Nothing below reports a number that is not already in [results.csv](../../results/results.csv) or reproduced by a script in this repo.

Notebook: [phase5-llm-prompting.ipynb](../../notebooks/phase5-llm-prompting.ipynb). Parser: [phase5_parser.py](../../scripts/phase5_parser.py), tested by [test_phase5_parser.py](../../scripts/test_phase5_parser.py). Gate: [phase5_gate.py](../../scripts/phase5_gate.py). Error analysis: [error_analysis.md](error_analysis.md).

The job of this phase: prompt a self-hosted open LLM (loaded in the notebook, no external API) for Hindi NER with zero-shot and few-shot prompts, turn its output into IOB2 tags, score it with the shared `src.evaluate`, and explain where it fails. This is a baseline; the expected finding is that it trails fine-tuned MuRIL (0.8274).

## Setup

- **Candidates (at most two, both fit a free T4):**

  | Key | Model | Load | License | Gated |
  | --- | ----- | ---- | ------- | ----- |
  | `qwen2.5-3b` | `Qwen/Qwen2.5-3B-Instruct` @ `aa8e72537993ba99e69dfaafa59ed015b17504d1` | fp16 | `qwen-research` | no |
  | `qwen2.5-7b-4bit` | `Qwen/Qwen2.5-7B-Instruct` @ `a09a35458c702b33eeacc393d103063234e8bc28` | 4-bit nf4 | Apache-2.0 | no |

  License, gating and revisions were read from the Hugging Face Hub. Airavata (Llama 2 license, gated), Aya Expanse 8B (CC-BY-NC, gated) and Navarasa (Gemma terms) were left out so the run needs no access approvals. The 3B model's `qwen-research` license limits it to research use, which fits a course project but is worth stating in the report.
- **Decoding:** greedy, batch size 8, `max_new_tokens=320`. A truncated answer has no closing bracket, so it counts as malformed.
- **Answer format** (same in both modes): `[{"text": "<entity as written>", "type": "PER|ORG|LOC"}]`, `[]` when there are no entities.
- **Few-shot:** 8 fixed examples (7 entity-rich, 1 with no entities) drawn from the Hindi **val** split. They sit outside the 150 validation sentences reserved for tuning, share no text with the gold test set, and cover PER, ORG and LOC. Their labels are silver, a limitation.
- **Scoring:** the shared `src.evaluate` (seqeval entity-level micro-F1, overall and per type). Final stage: **all 864** Hindi gold sentences, the same set as every other stage.
- **Traceability:** every row's `notes` carries the model revision, a hash of the prompt and shots, the parser's hash, a config fingerprint, the commit, library versions and the malformed count. The notebook saves raw outputs and per-sentence predictions under `results/phase5/`, and the gate re-scores them.

## How the output becomes IOB2 (the parser)

Rules R1 to R9 are written at the top of [phase5_parser.py](../../scripts/phase5_parser.py) and covered by 31 tests that need no GPU. The ones that decide scores:

| Situation | Rule |
| --------- | ---- |
| Empty text, no JSON, truncated or broken JSON, wrong shape | **Malformed:** all `O`, counted, never dropped. No partial rescue of a truncated list. |
| Prose or a code fence around valid JSON | Accepted, counted as format drift (`noisy`). |
| Item without `text`/`type`, or an unknown type | Skipped and counted (`invalid_items`, `unknown_label`). `PERSON`, `ORGANIZATION`, `LOCATION` map to PER, ORG, LOC. |
| Span text not in the sentence (translated or invented) | Skipped and counted (`unaligned_spans`). |
| Matching | Whole word tokens, comparing text after Unicode NFC with whitespace and zero-width joiners removed. This matters because Naamapadam splits punctuation off words (`एम .`) while a model writes `एम.`. |
| Same name twice in a sentence | **Every occurrence is tagged.** |
| Overlapping spans | Longest first, ties by listing order; a tagged token is never re-tagged. Same text with two types: first listed wins. |

**Parser ceiling.** Feeding the *gold* answers through the parser and scoring gives F1 **0.9984** on the Hindi gold test set (864 sentences) and 0.9928 on the full val set (13,460). So the parser itself costs under 0.2 F1 points on gold; everything below that is the model. The small loss comes from the "tag every occurrence" rule when gold tags only one of two identical mentions.

## Results

**PENDING (final stage, after the lead confirms Phase 4 is merged).**

| Setup | Overall | PER | ORG | LOC | Malformed / 864 | Unaligned spans |
| ----- | ------- | --- | --- | --- | --------------- | --------------- |
| Zero-shot | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| Few-shot (k=8) | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |

For context, from [results.csv](../../results/results.csv): BiLSTM-CRF 0.8038, MuRIL handoff 0.8274, IndicBERT 0.7839, all on the same 864 gold sentences.

## Validation-stage findings (model choice, prompt tuning, speed, VRAM)

**PENDING (validation Kaggle run).** To record here: validation F1 of both candidates in both modes, malformed rate, seconds per sentence and peak VRAM (to confirm 864 sentences x 2 modes fits one session), the model the selection rule picked, and the frozen `PROMPT_HASH`.

Selection rule, written before any run: among candidates whose few-shot malformed rate is at most 5%, take the highest few-shot validation F1; if several are within 0.01 F1 of the best, take the faster one. The validation sample is silver-labelled (like Phase 3's size selection), so it is a rough guide.

## Limits

- One prompt and one model are frozen per final run; the prompt was tuned only on the validation sample, never on gold.
- Few-shot examples carry silver labels, so they may teach some label noise.
- Greedy decoding, one run: no seed-to-seed spread is available for the LLM.
- Small differences between zero-shot and few-shot (about a point of F1) are within what 864 sentences can distinguish; large gaps are not.
- The comparison rows come from different training regimes (the LLM is not trained), so the gap says how far prompting is from fine-tuning, not which model is "better".

## Gate

**Not yet run on real results.** `uv run python scripts/phase5_gate.py` checks the parser tests, both rows (all 864 sentences, pinned revision, malformed counted, notes complete), one clean notebook execution of one commit, that the saved predictions re-score to the logged F1, and that these docs and the roadmap report the same numbers. The gate was exercised on a synthetic fixture (passes on consistent data; fails when a logged F1 is edited, when a malformed sentence is not all `O`, and when `test_n` is not 864).
