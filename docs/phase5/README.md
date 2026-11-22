# Phase 5: LLM Prompting Baseline

> **Status: final run done, gate passed.** Final gold results below come from one clean Kaggle execution of commit `f1d8cbd`; every number is in [results.csv](../../results/results.csv) or reproducible from [results/phase5/](../../results/phase5/).

Notebook: [phase5-llm-prompting.ipynb](../../notebooks/phase5-llm-prompting.ipynb). Parser: [phase5_parser.py](../../scripts/phase5_parser.py), tested by [test_phase5_parser.py](../../scripts/test_phase5_parser.py). Gate: [phase5_gate.py](../../scripts/phase5_gate.py). Error analysis: [error_analysis.md](error_analysis.md), with the cross-stage notebook [phase5-cross-stage-errors.ipynb](../../notebooks/phase5-cross-stage-errors.ipynb).

The job of this phase: prompt a self-hosted open LLM (loaded in the notebook, no external API) for Hindi NER with zero-shot and few-shot prompts, turn its output into IOB2 tags, score it with the shared `src.evaluate`, and explain where it fails. This is a baseline; the expected finding is that it trails fine-tuned MuRIL (0.8274).

## Setup

- **Candidates (at most two, both fit a free T4); the 7B model was chosen on validation:**

  | Key | Model | Load | License | Gated |
  | --- | ----- | ---- | ------- | ----- |
  | `qwen2.5-3b` | `Qwen/Qwen2.5-3B-Instruct` @ `aa8e72537993ba99e69dfaafa59ed015b17504d1` | fp16 | `qwen-research` | no |
  | `qwen2.5-7b-4bit` | `Qwen/Qwen2.5-7B-Instruct` @ `a09a35458c702b33eeacc393d103063234e8bc28` | 4-bit nf4 | Apache-2.0 | no |

  License, gating and revisions were read from the Hugging Face Hub. Airavata (Llama 2 license, gated), Aya Expanse 8B (CC-BY-NC, gated) and Navarasa (Gemma terms) were left out so the run needs no access approvals. The 3B model's `qwen-research` license limits it to research use, which fits a course project but is worth stating in the report.
- **Decoding:** greedy, batch size 8, `max_new_tokens=320`. An answer cut off at the limit is counted as `truncated`; it is malformed only if its JSON is incomplete (in validation, 1 zero-shot answer was truncated yet still parsed).
- **Answer format** (same in both modes): `[{"text": "<entity as written>", "type": "PER|ORG|LOC"}]`, `[]` when there are no entities.
- **Few-shot:** 8 fixed, hand-checked examples (7 entity-rich, 1 with no entities) from the Hindi **val** split (`SHOT_VAL_IDX` in the notebook). They sit outside the 150 validation sentences reserved for tuning, share no text with the gold test set, and cover PER, ORG and LOC. Their labels are silver, a limitation.
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

Hindi **gold** test, all **864** sentences, one clean Kaggle run (commit `f1d8cbd`, torch 2.11.0+cu130, transformers 5.22.1, greedy). Model `Qwen/Qwen2.5-7B-Instruct` (4-bit), revision `a09a35458c702b33eeacc393d103063234e8bc28`, prompt `v2/85343238`, config `798f7eb1`.

| Setup | Overall | PER | ORG | LOC | Malformed / 864 | Truncated | Unaligned spans |
| ----- | ------- | --- | --- | --- | --------------- | --------- | --------------- |
| Zero-shot | **0.6328** | 0.6397 | 0.5644 | 0.6812 | 11 (1.27%) | 6 | 32 |
| Few-shot (k=8) | **0.6643** | 0.6714 | 0.5956 | 0.7190 | 5 (0.58%) | 5 | 31 |
| MuRIL handoff (Phase 3, seed 42) | 0.8274 | 0.8879 | 0.7444 | 0.8224 | n/a | n/a | n/a |
| BiLSTM-CRF (Phase 2) | 0.8038 | 0.8692 | 0.7130 | 0.7933 | n/a | n/a | n/a |
| IndicBERT (Phase 3) | 0.7839 | 0.8538 | 0.6968 | 0.7692 | n/a | n/a | n/a |

Malformed counts: zero-shot malformed: 11, few-shot malformed: 5. Both rows are in [results.csv](../../results/results.csv) with their full setup in `notes`. Phase 4 numbers are not used here.

## Findings

- **The LLM trails every fine-tuned model,** as expected. Few-shot (0.6643) is 0.163 F1 below MuRIL, 0.140 below the BiLSTM-CRF baseline and 0.120 below IndicBERT. That gap is the finding of this stage.
- **Examples help: +0.0315 overall** (0.6328 to 0.6643), with gains on all three types (PER +0.032, ORG +0.031, LOC +0.038). This is indicative only: it comes from one model, one prompt and one greedy run, so no spread was measured.
- **ORG is the weakest type** for the LLM too (0.5956 few-shot), as for every other model. Against MuRIL the gap is largest on PER (−0.217), then ORG (−0.149) and LOC (−0.103).
- **Format drift is small.** Malformed outputs were 11 of 864 for zero-shot and 5 of 864 for few-shot. Six zero-shot and all five few-shot failures were answers cut off at the 320-token limit. The other five zero-shot failures were short answers with broken JSON, so a larger token limit would not fix them. Every malformed output was scored as all `O`. No output was wrapped in prose or fences. 32 and 31 spans did not match the sentence (altered spelling, an added word, stray characters) and were lost.
- **Where it fails** ([error_analysis.md](error_analysis.md)): about twice MuRIL's errors (1,003 vs 480). Boundary errors are mostly over-long spans (a role or title attached to a name; 276 of 317 are longer than gold); false alarms are generic words (`देश`, `प्रदेश`, `राष्ट्रपति`, `पुलिस`); LOC is often read as ORG, which MuRIL also does, so part of that is gold conventions.

## Validation-stage findings (model choice, prompt tuning, speed, VRAM)

Validation sample: 150 Hindi **val** sentences (silver labels, seed 42), reserved so no few-shot example comes from them. These F1 values are comparable with each other, not with gold scores.

**Round 1: both candidates, prompt v1 (8 randomly sampled silver shots).**

| Model | Mode | F1 | PER | ORG | LOC | Malformed | sec/sent | Est. min for 864 | Peak VRAM |
| ----- | ---- | -- | --- | --- | --- | --------- | -------- | ---------------- | --------- |
| qwen2.5-3b | zero-shot | 0.4741 | 0.5616 | 0.3141 | 0.5196 | 3/150 | 0.76 | 11 | 6.6 GB |
| qwen2.5-3b | few-shot | 0.4429 | 0.5078 | 0.2717 | 0.5333 | 2/150 | 1.62 | 23 | 7.6 GB |
| qwen2.5-7b-4bit | zero-shot | 0.5447 | 0.6076 | 0.3944 | 0.6090 | 0/150 | 2.80 | 40 | 6.8 GB |
| qwen2.5-7b-4bit | few-shot | 0.5545 | 0.6188 | 0.3390 | 0.6529 | 2/150 | 5.71 | 82 | 11.9 GB |

Both few-shot malformed rates (1.3%) met the 5% limit, and the 7B model's few-shot F1 was 0.11 higher, so the selection rule picked **`qwen2.5-7b-4bit`** (Apache-2.0, ungated).

**Round 2: 7B only, prompt v2 (8 hand-checked val shots).** Round 1 showed some silver shots had wrong labels (for example a film title tagged PER), so v2 replaces them with 8 hand-checked sentences (`SHOT_VAL_IDX`).

| Mode | F1 | PER | ORG | LOC | Malformed | Truncated | Unaligned | sec/sent | Est. min for 864 | Peak VRAM |
| ---- | -- | --- | --- | --- | --------- | --------- | --------- | -------- | ---------------- | --------- |
| zero-shot | 0.5447 | 0.6076 | 0.3944 | 0.6090 | 0/150 | 1 | 6 | 2.60 | 37 | 6.8 GB |
| few-shot v2 | 0.5523 | 0.6460 | 0.3333 | 0.6508 | 0/150 | 0 | 8 | 5.23 | 75 | 11.9 GB |

v2 did not change overall F1 (0.5545 to 0.5523, within noise on 150 sentences). PER rose by about 0.03 and malformed and truncated outputs went to zero. ORG stayed near 0.33, below zero-shot ORG (0.394), in both rounds. Parser ceiling on this sample: 0.9853.

**Decision (frozen before the final gold run):** model `qwen2.5-7b-4bit`, prompt `v2`, `FINAL_MODEL` set in the notebook. Tuning stopped here because further changes on a noisy silver sample risk fitting its noise. The final run needs about 1.9 hours (37 + 75 minutes) and 11.9 GB VRAM, which fits one Kaggle session. Prompt `v2/85343238`; parser hash `ffb1d6e7` (from the Kaggle clone) in the validation log.

Selection rule, written before any run: among candidates whose few-shot malformed rate is at most 5%, take the highest few-shot validation F1; if several are within 0.01 F1 of the best, take the faster one.

## Limits

- **Gold test exposure.** An earlier 200-sentence LLM draft (a 3B model) was run on gold test sentences before this phase's final setup was built. It was retired because it was unpinned and untraceable, and its scores are not reported anywhere. It did influence the final setup in these ways: the `Qwen2.5-3B-Instruct` candidate, the instruction wording "copy each entity exactly as written, same script, same spelling, no translation" (adapted), the choices of 8 few-shot examples and batch size 8, and the code that sorts errors into missed, boundary, type confusion and spurious. The draft's gold scores had also been seen before the final setup was built. So the final setup is not fully independent of the gold test set. What was done independently of gold: the final model (7B) and prompt `v2` were selected on a silver validation sample, and the gold test set was scored once, with everything frozen.
- One prompt and one model are frozen per final run; within this phase's own runs, the model and prompt were tuned only on the validation sample.
- Few-shot examples carry silver labels, so they may teach some label noise.
- Greedy decoding, one run: no seed-to-seed spread is available for the LLM.
- The zero-shot to few-shot gain (+0.03) has no measured spread. The 0.12 to 0.16 gaps to the fine-tuned models are far larger than that and than the 0.01 differences seen between prompt versions on validation.
- **Malformed outputs.** Malformed outputs were 11 of 864 for zero-shot and 5 of 864 for few-shot. Six zero-shot and all five few-shot failures were answers cut off at the 320-token limit. The other five zero-shot failures were short answers with broken JSON, so a larger token limit would not fix them. Four gold sentences need more than 320 tokens for a fully correct answer (checked with the Qwen tokenizer on the gold answers). A larger limit was not tested, and it should not be rerun now, because that would change the setup after seeing gold scores.
- **Parser.** It can accept a valid JSON prefix of an answer that was cut off, and a list with no entity objects, such as `[1]`, is not counted as malformed (its items are only counted as invalid). Neither case changed the committed scores.
- **Error analysis.** When one gold entity overlaps several predicted fragments, only the first is counted, so boundary errors are slightly undercounted.
- **Notebook, for future reruns.** The output cache key omits `MAX_NEW_TOKENS`, outputs are saved only after a whole mode finishes, and the model-speed tie-break treats a cached timing as 0 seconds. None of this affected the committed run, which started with an empty cache.
- The comparison rows come from different training regimes (the LLM is not trained), so the gap says how far prompting is from fine-tuning, not which model is "better".

## Gate

**Passed.** `uv run python scripts/phase5_gate.py` checks the 31 parser tests, both rows (all 864 sentences, pinned revision, malformed counted, notes complete), one clean notebook execution of one commit (every cell ran once in order, no errors, printed rows equal the logged rows), that the saved predictions re-score to the logged F1 with every malformed sentence all `O`, and that these docs and the roadmap report the same numbers. Before the real run it was exercised on a synthetic fixture: it passed on consistent data and failed when a logged F1 was edited, when a malformed sentence was not all `O`, and when `test_n` was not 864.

## Artifacts

- **Notebook:** [phase5-llm-prompting.ipynb](../../notebooks/phase5-llm-prompting.ipynb), the executed Kaggle version, committed unchanged.
- **Per-sentence results:** [final_predictions.jsonl](../../results/phase5/final_predictions.jsonl) (raw model output, parse flags and tags for all three systems, 864 sentences each) and [final_errors.csv](../../results/phase5/final_errors.csv).
- **Model:** [Qwen/Qwen2.5-7B-Instruct @ a09a354](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/tree/a09a35458c702b33eeacc393d103063234e8bc28), Apache-2.0, not gated.
- **Earlier draft:** a first 200-sentence draft with a 3B model was run on gold test sentences and retired before this phase's final run (unpinned, untraceable). Its scores are not reported anywhere; see Limits for what carried over from it.
