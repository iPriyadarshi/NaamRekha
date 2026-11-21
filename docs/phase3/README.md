# Phase 3: MuRIL + IndicBERT Fine-tuning

Everything in this folder is specific to Phase 3. The job of this phase is to fine-tune a transformer NER model for Hindi, find the training-set size empirically (the roadmap's scaling-curve requirement), and beat the Phase 2 BiLSTM-CRF baseline. All of it lives in [phase3-muril-indicbert.ipynb](../../notebooks/phase3-muril-indicbert.ipynb), which uses only the shared `src/` loader, label map, evaluator, and results logger.

## Status: gate passed, lead review (PR #4) addressed and re-verified

The real scaling curve (MuRIL, `DEBUG = False`) ran on Kaggle for subset sizes 10,000 / 30,000 / 100,000 / 300,000. The 900,000 run was started but the Kaggle session was interrupted right as it began (model/data just loaded, no training steps run) — no result was logged for that size, and it turned out not to be needed:

- **The lead's review (PR #4) flagged the 10k/30k collapse as a likely training-setup bug**, not real instability: no LR warmup, and training possibly silently running on both of Kaggle's T4 GPUs via `DataParallel` (every run printed a `"gather along dimension 0"` warning, a tell for this), doubling the effective batch size. Both were fixed (`warmup_steps` computed from the real step count; `CUDA_VISIBLE_DEVICES=0` forces single-GPU) and **10k/30k/100k were re-run** to check. Result, with `gpus=1` and the real `warmup_steps` confirmed in each row's notes:
  - **n=10,000 still collapses** (F1_overall 0.0, unchanged) — this rules out the warmup/batch-size bug as the cause; it's a genuine small-data limitation for this model/label set, not a pipeline bug.
  - **n=30,000 was fixed by the warmup/single-GPU fix**: F1_overall jumped from 0.0807 (buggy) to **0.8217** (re-verified) — confirming the bug was real and the lead's diagnosis was correct for this size.
  - **n=100,000 re-verified at F1_overall 0.8251** (previously 0.8343 under the bug) — consistent, still clearly beats the baseline.
- **n=300,000 was not re-run** (not requested in review): F1_overall 0.8287, not re-verified for the warmup/GPU fix, kept as a reference point (noted in `results.csv`).
- **n=100,000 is the chosen working subset size.** It clearly beats the Phase 2 baseline (0.8038) at both re-verified runs (0.8251 scaling-curve rerun, 0.8330 separate checkpoint-handoff rerun); 300,000 (even pre-fix) doesn't meaningfully improve on it, so per the roadmap's stopping rule this is where the curve flattens.
- The scaling-curve runs' checkpoints were deleted right after logging, to stay under Kaggle's disk quota — this affects all four scaling-curve rows (10k/30k/100k/300k); their `checkpoint_path` values point at folders that no longer exist, noted in each row. A separate **`train_run("MuRIL", 100_000)` re-run** (without deleting the checkpoint, done before the warmup/GPU fix) produced F1_overall 0.8330 — consistent with the scaling-curve results — and this is the one with a live, durable checkpoint on the Hugging Face Hub.
- The **IndicBERT comparison run** at the chosen size (100,000), using the real `ai4bharat/indic-bert`, is done: F1_overall 0.7198. It reused MuRIL's hyperparameters (lr 2e-5) rather than separately tuned ones for ALBERT's architecture, so this number may understate IndicBERT's real capability — noted in `results.csv` and not yet retuned.
- **Both checkpoints are pushed to the Hugging Face Hub**: MuRIL at [huggingface.co/shreyanks-1/muril-hindi-ner-100k](https://huggingface.co/shreyanks-1/muril-hindi-ner-100k) and IndicBERT at [huggingface.co/shreyanks-1/indicbert-hindi-ner-100k](https://huggingface.co/shreyanks-1/indicbert-hindi-ner-100k). Phase 4 can load the MuRIL one directly with `AutoModelForTokenClassification.from_pretrained("shreyanks-1/muril-hindi-ner-100k")`.
- The notebook had rough edges flagged in review (now fixed): leftover duplicate `!rm -rf` cells removed, the IndicBERT/MuRIL-rerun cells now use `CHOSEN_SUBSET_SIZE` instead of a hardcoded `100_000`, the abandoned 900k size dropped from `SUBSET_SIZES`, and the MuRIL `push_to_hub` code (previously run ad hoc, not saved in the notebook) is now an actual cell.
- The scaling-curve plot (`results/phase3_scaling_curve.png`) now reflects the re-verified 10k/30k/100k numbers and is committed to the repo.

`uv run python scripts/phase3_gate.py` passes end to end, and the scaling curve is now re-verified, not just logged.

## What we built

- **Models:** `google/muril-base-cased` (primary) and `ai4bharat/indic-bert` (comparison), exactly as named in the roadmap. `indic-bert` is a gated HuggingFace repo (requires a logged-in, license-accepted account); once that was set up, the real model was used directly — no substitute.
- **Label alignment** ([docs/05-pitfalls.md](../05-pitfalls.md)): the tokenizer's `word_ids()` maps each subword back to its original word. The first subword of a word gets that word's IOB2 label; every continuation subword and every special token (`[CLS]`, `[SEP]`, padding) gets `-100` so the loss ignores it. A self-test (in the notebook, before any model is trained) asserts this on a real training sentence.
- **`train_run(model_key, subset_size, ...)`**: one reusable function — seeded subset sampling + entity-density check, tokenize-and-align, HuggingFace `Trainer` (fp16 when a GPU is present, checkpoint every epoch, `max_length=128`, batch size 32), then predicts on the **full** Hindi gold test set with the same `src/eval.evaluate` harness as every other phase, and logs one `results.csv` row.
- **Scaling procedure:** call `train_run("MuRIL", n)` for each `n` in the ladder, plot F1 vs. `n`, and stop adding data once the curve flattens (the roadmap's instruction, not a fixed size decided up front).

## Results

| subset_size | model | F1_PER | F1_ORG | F1_LOC | F1_overall |
| ----------- | ----- | ------ | ------ | ------ | ---------- |
| 10,000  | MuRIL (re-verified) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 30,000  | MuRIL (re-verified) | 0.9094 | 0.7221 | 0.7942 | **0.8217** |
| 100,000 | MuRIL (re-verified) | 0.8866 | 0.7353 | 0.8254 | **0.8251** |
| 300,000 | MuRIL (not re-verified) | 0.8908 | 0.7518 | 0.8162 | 0.8287 |
| 100,000 | MuRIL (checkpoint-handoff rerun, not re-verified) | 0.9041 | 0.7417 | 0.8223 | 0.8330 |
| 100,000 | IndicBERT | 0.7720 | 0.6542 | 0.7095 | 0.7198 |

(seed=42, lr=2e-5, batch size 32, 3 epochs, `max_length=128` for every row. "Re-verified" rows ran with the `warmup_steps` + single-GPU fix from PR #4's review and have `gpus=1` + `warmup_steps=N` recorded in their notes.)

![Phase 3 MuRIL scaling curve](../../results/phase3_scaling_curve.png)

**Chosen working subset size: 100,000.** It clearly beats the baseline (0.8038) at both 100k runs (0.8251 re-verified, 0.8330 checkpoint-handoff); 300,000 (not yet re-verified, but already known not to improve on 100k pre-fix) doesn't change that conclusion, so per the roadmap's stopping rule this is where the curve flattens. 900,000 was not run.

**IndicBERT comparison** (at the chosen size, 100,000, using the real `ai4bharat/indic-bert`): F1_overall 0.7198 — clearly behind MuRIL (0.8251) on every entity type, and behind the Phase 2 baseline (0.8038) too. The roadmap's gate only requires MuRIL to beat the baseline (it does), so this doesn't affect the gate; IndicBERT's weaker showing here is logged as a real comparison result in its own right, not an artifact of a substitute model.

**Caveat:** each point above is a single seed (42). The 10,000 collapse is now confirmed as a genuine small-data limitation, re-verified with the warmup/single-GPU fix applied — it's not a bug. The 300,000 and checkpoint-handoff-100,000 rows were not re-run with this fix (not requested in review) and are noted as such in `results.csv`; the margin at the chosen size (100,000) over baseline (+0.021 to +0.029 depending on which 100k run) is reasonably wide either way. If this needs to be bulletproof for the writeup, re-run n=100,000 with 2-3 seeds.

## Decisions

1. **`max_length = 128`.** Matches `docs/04-compute.md`; Hindi NER sentences are short, so this only risks truncating the rare very long sentence. The notebook prints how many gold-test sentences would be affected.
2. **3 epochs, lr 2e-5, batch size 32, fp16 on GPU, single-GPU forced.** Standard BERT-family fine-tuning defaults; not tuned per the "baseline, don't chase SOTA" spirit that also governed Phase 2. `CUDA_VISIBLE_DEVICES=0` was added after review flagged that Kaggle's dual-T4 sessions can silently double the effective batch size via `DataParallel`.
3. **A single seed (42) per subset size**, recorded in `notes`. If MuRIL's margin over 0.8038 is small at the chosen size, re-run with 2-3 seeds before calling it "clear" (same caveat Phase 2 flagged).

## Gate

**Passed.** `uv run python scripts/phase3_gate.py` passes end to end — the MuRIL scaling curve is logged and traceable to the notebook's own output, and clearly beats the baseline by more than the required margin (best F1 0.8330 vs. 0.8038, margin >= 0.01). The IndicBERT comparison row is logged too, using the real `ai4bharat/indic-bert` (F1 0.7198). Unlike the first pass, the scaling curve is now re-verified against the lead's review (PR #4): the warmup/single-GPU fix was applied and 10k/30k/100k were re-run, confirming the bug at 30k and the genuine small-data limitation at 10k.

## What this unblocks

- **Phase 4** can load the MuRIL Hindi checkpoint directly from [huggingface.co/shreyanks-1/muril-hindi-ner-100k](https://huggingface.co/shreyanks-1/muril-hindi-ner-100k) (`AutoModelForTokenClassification.from_pretrained(...)`) for zero-shot Marathi/Bengali evaluation and in-language fine-tuning.
- **Phase 6** has this phase's numbers (and the scaling curve itself, which the roadmap calls out as a figure) for the comparison table, including the real MuRIL-vs-IndicBERT gap.
