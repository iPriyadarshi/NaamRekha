# Phase 3: MuRIL + IndicBERT Fine-tuning

Everything in this folder is specific to Phase 3. The job of this phase was to fine-tune a transformer NER model for Hindi, find the working training size from evidence, and beat the Phase 2 BiLSTM-CRF baseline (0.8038). All of it lives in [phase3-muril-indicbert.ipynb](../../notebooks/phase3-muril-indicbert.ipynb), which uses only the shared `src/` loader, label map, evaluator, and results logger.

**Result:** MuRIL at 300,000 sentences scores Hindi gold entity F1 **0.8274** (seed 42, the handoff model). Its mean across three seeds is **0.8230**, and all three seeds beat the baseline.

**Chosen working subset size: 300,000.**

## What we built

- **Models:** `google/muril-base-cased` (primary) and `ai4bharat/indic-bert` (comparison).
- **Label alignment:** each word's label goes on its first subword. Other subwords and special tokens get `-100`. IndicBERT's tokenizer can emit a standalone `▁` as a word's first piece, so the label goes on the first real piece after it. This affected 1,254 of 43,423 sampled IndicBERT words and 0 MuRIL words.
- **Evaluation:** the shared `src.eval.evaluate` harness on the full gold test set (864 sentences). Words lost to truncation stay in the gold labels and count as predicted `O`. This affects 0 MuRIL and 4 IndicBERT test sentences.
- **Training run:** one reusable `train_run()` handles subset sampling, training, validation and gold scoring, checkpoint upload, and logging.
- **Checkpoints:** every run uploads its final model, confirms the upload, logs a link pinned to that Hugging Face revision, and only then deletes the local copy.

## Setup

All 13 rows came from one top-to-bottom Kaggle execution of commit `77b5e99`, with torch 2.11.0 (CUDA 13.0) and `transformers` 5.22.1.

- **Training:** silver subsets, lr 2e-5, batch size 32, 3 epochs unless stated, `max_length=128`, fp16, and linear warmup over the first 10% of steps.
- **GPU:** exactly one T4. Kaggle's second GPU is hidden so `DataParallel` cannot double the effective batch size.
- **Size selection:** the smallest size with no collapsed seed whose mean **validation** F1 is within 0.01 of the best size. The rule was set before the run. Validation is the full silver set (13,460 sentences).
- **Reporting:** gold test F1 was not used to choose the size.

## Results

Scaling curve, MuRIL, 3 epochs:

| Subset size | Seeds | Steps per run | Val F1 mean (range) | Gold F1 mean (range) |
| ----------- | ----- | ------------- | ------------------- | -------------------- |
| 10,000 | 1 | 939 | 0.0000 | 0.0000 |
| 30,000 | 3 | 2,814 | 0.7509 (0.7481 to 0.7539) | 0.8244 (0.8182 to 0.8285) |
| 100,000 | 3 | 9,375 | 0.7816 (0.7808 to 0.7825) | 0.8296 (0.8266 to 0.8345) |
| **300,000** | 3 | 28,125 | **0.7921** (0.7913 to 0.7927) | **0.8230** (0.8174 to 0.8274) |

Handoff and comparison runs on Hindi gold (seed 42, 300,000 sentences):

| Model | PER | ORG | LOC | Overall |
| ----- | --- | --- | --- | ------- |
| MuRIL (handoff) | 0.8879 | 0.7444 | 0.8224 | **0.8274** |
| IndicBERT (untuned) | 0.8538 | 0.6968 | 0.7692 | 0.7839 |
| BiLSTM-CRF baseline (Phase 2, 100,000) | 0.8692 | 0.7130 | 0.7933 | 0.8038 |

Diagnostic runs (MuRIL, seed 42):

| Run | Steps | Val F1 | Gold F1 | What it shows |
| --- | ----- | ------ | ------- | ------------- |
| 10,000, 9 epochs | 2,817 | 0.7567 | 0.8231 | The 3-epoch collapse was under-training |
| 30,000, 10 epochs | 9,380 | 0.7685 | 0.8180 | More epochs did not improve gold F1 at 30,000 |

The curve is in [phase3_scaling_curve.png](../../results/phase3_scaling_curve.png). Every row is in [results.csv](../../results/results.csv).

## Findings

- **MuRIL beats the baseline on every seed at the chosen size.** The margin is +0.0236 for the handoff model, and the lowest seed (0.8174) is still +0.0136. The baseline itself has one seed.
- **The 10,000 collapse was not a data limit.** At 3 epochs the model predicted no entities. With 9 epochs and everything else unchanged, it reached gold F1 0.8231.
- **The size choice was close.** 100,000 missed the validation cutoff by 0.0005 (mean 0.7816 against a cutoff of 0.7821). Every 300,000 seed beat every 100,000 seed on validation.
- **Gold test does not separate the sizes.** 100,000 has the higher gold mean (0.8296 against 0.8230), but that 0.0066 gap is about the size of the seed-to-seed variation on an 864-sentence test set. Validation is larger but uses silver labels, so it may reward fitting silver noise.
- **Gold F1 is flat from 30,000 upward.** All three mean scores fall between 0.8230 and 0.8296. A smaller size may work nearly as well if compute becomes the limiting factor.
- **ORG is still the weakest type** for every model, so it is the first place to look in Phase 5's error analysis.

## Decisions

1. **Keep 300,000 as the working size.** It is what the pre-set validation rule selected. Switching to 100,000 because of its gold score would use the test set for model selection.
2. **Hand off seed 42.** It is the pre-set handoff seed, not the best gold score picked after the run. Report the three-seed mean next to its score.
3. **IndicBERT stays an untuned comparison.** It reuses MuRIL's hyperparameters on one seed, so 0.7839 is not its best possible score.
4. **Earlier PR #4 runs are retired.** Phase 3 was restarted from a clean state. Those runs remain in Git history, but they are not results of this phase.

## Artifacts

- **Notebook:** [phase3-muril-indicbert.ipynb](../../notebooks/phase3-muril-indicbert.ipynb), committed with its outputs from the Kaggle run.
- **Phase 4 handoff checkpoint:** [`priyadarshi-su/naamrekha-muril_hi_n300000_e3_lr2e-05_s42_bfd254ed`](https://huggingface.co/priyadarshi-su/naamrekha-muril_hi_n300000_e3_lr2e-05_s42_bfd254ed/tree/de5204680ed4a8ab1ac842847df6866a6e2c9251), revision `de5204680ed4a8ab1ac842847df6866a6e2c9251`. It was reloaded from this revision and re-scored at 0.8274.
- **IndicBERT checkpoint:** [`priyadarshi-su/naamrekha-indicbert_hi_n300000_e3_lr2e-05_s42_3b287f89`](https://huggingface.co/priyadarshi-su/naamrekha-indicbert_hi_n300000_e3_lr2e-05_s42_3b287f89/tree/baacb700879d413a7712f4767398599154535d5c).
- **Access:** every checkpoint linked from `results.csv` is public, so loading it needs no token.

## Gate

**Passed.** `uv run python scripts/phase3_gate.py` checks that every row comes from one clean execution of one commit, uses the fixed setup, and has a pinned checkpoint. It also checks the size rule, the three seeds, the handoff re-score, the baseline margin, the IndicBERT row, and that the docs agree.

## What this unblocks

- **Phase 4** can start from the pinned MuRIL revision above.
- **All later phases** use 300,000 as the working subset unless the team records a new decision.
- **Phase 5** has ORG as the first error-analysis target.
- **Phase 6** has the scaling curve and the model comparison table.
