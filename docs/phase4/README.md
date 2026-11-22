# Phase 4: Cross-lingual Transfer

Everything in this folder is specific to Phase 4. The job of this phase was to measure how well a Hindi-trained MuRIL NER model transfers to Marathi (same Devanagari script) and Bengali (different script), compared with models trained in each language. All of it lives in [phase4-transfer.ipynb](../../notebooks/phase4-transfer.ipynb), which uses only the shared `src/` loader, label map, evaluator, and results logger.

**Result:** across three seeds, the Hindi model scores mean gold F1 **0.8332** on Marathi and **0.8037** on Bengali with no target-language training. The in-language baselines score **0.8163** and **0.8132**. Hindi beats the Marathi-trained model on all three seeds, and the Bengali-trained model beats Hindi on all three seeds.

## What we built

- **Zero-shot:** three pinned Phase 3 Hindi models (seeds 42, 43, and 44) scored on the full `hi`, `mr`, and `bn` gold test sets, with no further training.
- **In-language baselines:** `google/muril-base-cased` fine-tuned on 300,000 `mr` sentences and 300,000 `bn` sentences with the Phase 3 recipe, once per seed.
- **Transfer gap:** in-language F1 − zero-shot F1, paired by seed, per target language and entity type. A negative gap means the Hindi model scored higher.
- **Label alignment and scoring:** the same code as Phase 3. Words lost to truncation count as predicted `O`.
- **Precision:** every Phase 4 score comes from a freshly loaded model in full 32-bit, for both zero-shot and in-language runs.
- **Checkpoints:** each in-language run uploaded its final model, confirmed the upload, logged a link pinned to that revision, and only then deleted the local copy.

## Setup

All 15 rows came from one top-to-bottom Kaggle execution of commit `1dab156`, with torch 2.11.0 (CUDA 13.0) and `transformers` 5.22.1. The run took 6.63 hours.

- **Training:** seeds 42, 43, and 44, lr 2e-5, batch size 32, 3 epochs, `max_length=128`, fp16, warmup over the first 10% of steps, and exactly one T4 GPU.
- **Silver train sets:** 455,248 Marathi and 961,679 Bengali sentences, each sampled to 300,000 per seed.
- **Gold test sets:** 864 Hindi, 1,060 Marathi, and 607 Bengali sentences.
- **Hindi reproduction check:** each model had to score within 0.001 of its Phase 3 gold F1. Seeds 42 and 44 matched exactly. Seed 43 scored 0.8237 against 0.8242, because Phase 3 scored it under fp16 autocast right after training, while Phase 4 scores a reloaded model in 32-bit.

## Results

Mean gold entity F1 across seeds (± standard deviation):

| train \ test | hi | mr | bn |
| ------------ | -- | -- | -- |
| hi | **0.8228** ± 0.0050 | **0.8332** ± 0.0030 | **0.8037** ± 0.0055 |
| mr | | **0.8163** ± 0.0007 | |
| bn | | | **0.8132** ± 0.0030 |

Overall F1 per seed:

| Train → test | Seed 42 | Seed 43 | Seed 44 |
| ------------ | ------- | ------- | ------- |
| hi → hi | 0.8274 | 0.8237 | 0.8174 |
| hi → mr (zero-shot) | 0.8298 | 0.8355 | 0.8343 |
| mr → mr (in-language) | 0.8156 | 0.8170 | 0.8164 |
| hi → bn (zero-shot) | 0.7982 | 0.8036 | 0.8091 |
| bn → bn (in-language) | 0.8136 | 0.8100 | 0.8159 |

Mean F1 by entity type:

| Train → test | PER | ORG | LOC |
| ------------ | --- | --- | --- |
| hi → hi | 0.8867 | 0.7323 | 0.8210 |
| hi → mr | 0.8989 | 0.6883 | 0.8525 |
| mr → mr | 0.8884 | 0.6519 | 0.8432 |
| hi → bn | 0.8505 | 0.7180 | 0.7923 |
| bn → bn | 0.8585 | 0.7345 | 0.7983 |

Transfer gap, paired by seed (in-language − zero-shot):

| Target | Type | Mean gap | SD | Seed 42 | Seed 43 | Seed 44 |
| ------ | ---- | -------- | -- | ------- | ------- | ------- |
| Marathi | **Overall** | **−0.0169** | 0.0023 | −0.0142 | −0.0186 | −0.0178 |
| Marathi | PER | −0.0105 | 0.0028 | −0.0073 | −0.0122 | −0.0121 |
| Marathi | ORG | −0.0364 | 0.0071 | −0.0417 | −0.0391 | −0.0283 |
| Marathi | LOC | −0.0093 | 0.0076 | −0.0020 | −0.0089 | −0.0171 |
| Bengali | **Overall** | **+0.0095** | 0.0051 | +0.0154 | +0.0063 | +0.0068 |
| Bengali | PER | +0.0080 | 0.0065 | +0.0114 | +0.0004 | +0.0121 |
| Bengali | ORG | +0.0165 | 0.0093 | +0.0270 | +0.0127 | +0.0097 |
| Bengali | LOC | +0.0060 | 0.0079 | +0.0113 | +0.0098 | −0.0030 |

The heatmap is in [phase4_transfer_matrix.png](../../results/phase4_transfer_matrix.png). Every row is in [results.csv](../../results/results.csv).

## Findings

- **Hindi transfers strongly to both languages.** With no target-language training, its mean gold F1 is 0.8332 on Marathi and 0.8037 on Bengali.
- **On Marathi, the Hindi model beats the in-language baseline.** The mean gap is −0.0169, with the same sign on all three seeds. The Marathi run is therefore a baseline and not a true upper bound on this test set.
- **On Bengali, in-language training helps slightly.** The mean gap is +0.0095, also with the same sign on all three seeds. The Bengali LOC gap changes sign, so it is not established.
- **The pattern fits the shared-script idea, but does not prove it.** Hindi transfers better relative to the in-language baseline on Marathi than on Bengali. With only two target languages and different test sets, we cannot isolate script as the cause.
- **The reason for the Marathi result is unknown.** Differences in silver label quality are one possibility, but this phase did not test them.
- **ORG has the largest gaps** on both target languages and is the weakest type in every row, so it is the first target for Phase 5's error analysis.

## Limits

- The seeds measure training variation. Each language still has one small gold test set, and test-set sampling uncertainty was not measured.
- Absolute F1 should not be compared directly across target languages, because each language has a different test set.
- The three seeds also change the sampled training subset, so the reported spread includes both subset and training variation.

## Decisions

1. **Use three seeds.** The transfer gaps were too small to interpret from one seed.
2. **Report the `mr` and `bn` runs as in-language baselines.** The roadmap calls them upper bounds, but the Marathi result shows that is not guaranteed.
3. **Allow a 0.001 Hindi reproduction tolerance.** It covers the Phase 3 fp16 versus Phase 4 32-bit scoring difference. Phase 4 logs the score it measured and does not change Phase 3's rows.
4. **Use the same subset size and recipe as Phase 3,** so training language is the only planned difference between zero-shot and in-language models.

## Artifacts

- **Notebook:** [phase4-transfer.ipynb](../../notebooks/phase4-transfer.ipynb), committed with its outputs from the Kaggle run.

| Seed | Hindi | Marathi | Bengali |
| ---- | ----- | ------- | ------- |
| 42 | [model](https://huggingface.co/priyadarshi-su/naamrekha-muril_hi_n300000_e3_lr2e-05_s42_bfd254ed/tree/de5204680ed4a8ab1ac842847df6866a6e2c9251) | [model](https://huggingface.co/priyadarshi-su/naamrekha-muril_mr_n300000_e3_lr2e-05_s42_36701790/tree/25c48dfb02801f392faebcc0c0007cbd72a5cc73) | [model](https://huggingface.co/priyadarshi-su/naamrekha-muril_bn_n300000_e3_lr2e-05_s42_46e8e677/tree/f6171a437c97e61727b3c8710e4e4b4c31514a97) |
| 43 | [model](https://huggingface.co/priyadarshi-su/naamrekha-muril_hi_n300000_e3_lr2e-05_s43_223b5780/tree/f15d615b7d760d23af6c5aa111675f0d6be39c3d) | [model](https://huggingface.co/priyadarshi-su/naamrekha-muril_mr_n300000_e3_lr2e-05_s43_45139850/tree/f6e6b50e951499cccfd72d15198f34b8bdf68428) | [model](https://huggingface.co/priyadarshi-su/naamrekha-muril_bn_n300000_e3_lr2e-05_s43_19a22525/tree/d48a1f73c3686cd94785f41dc9da55f7b4c20a02) |
| 44 | [model](https://huggingface.co/priyadarshi-su/naamrekha-muril_hi_n300000_e3_lr2e-05_s44_618532f6/tree/aa2f28136b49859a460217190b85ebedfc625e32) | [model](https://huggingface.co/priyadarshi-su/naamrekha-muril_mr_n300000_e3_lr2e-05_s44_99872a74/tree/f485f37597e69d8a942e43a94301cc2df4177de8) | [model](https://huggingface.co/priyadarshi-su/naamrekha-muril_bn_n300000_e3_lr2e-05_s44_e7b8f845/tree/1d47716a7955c090b1d72b5fbc48b7f218605a76) |

## Gate

**Passed.** `uv run python scripts/phase4_gate.py` checks that the notebook ran once from a fresh kernel, and that all 15 rows (five matrix cells on three seeds) are logged once each from one code commit and match its output. It also checks the pinned Hindi checkpoints, that their scores are within 0.001 of Phase 3, the 300,000 size and single-GPU setup, and that this doc reports each cell's mean.

## What this unblocks

- **Phase 5** can run its final LLM evaluation after this phase merges, and can extend error analysis across all three languages, starting with ORG.
- **Phase 6** has the transfer matrix, paired gaps, heatmap, and limits for the script-vs-family discussion.
