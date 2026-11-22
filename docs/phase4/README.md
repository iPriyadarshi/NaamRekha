# Phase 4: Cross-lingual Transfer

Everything in this folder is specific to Phase 4. The job of this phase is to measure how well a Hindi-trained MuRIL NER model transfers to Marathi (same Devanagari script) and Bengali (different script), compared with models trained in each language. All of it lives in [phase4-transfer.ipynb](../../notebooks/phase4-transfer.ipynb), which uses only the shared `src/` loader, label map, evaluator, and results logger.

## Status: ready to run

The notebook, the gate script, and the matrix/heatmap code are in place, using the three final Phase 3 Hindi checkpoints, working size, and training recipe. No results are logged yet.

To finish:

1. Commit and push `phase4-cross-lingual`.
2. Run the notebook on Kaggle with `DEBUG = True`, then with `DEBUG = False`. The full run takes about 6.7 hours.
3. Add the results to this file, then run `uv run python scripts/phase4_gate.py`.

## Design

- **Seeds:** 42, 43, and 44. Results are reported as a mean and standard deviation across seeds.
- **Hindi checkpoints:** the three Phase 3 MuRIL models at n=300,000, each pinned to its revision:
  - Seed 42: [`priyadarshi-su/naamrekha-muril_hi_n300000_e3_lr2e-05_s42_bfd254ed`](https://huggingface.co/priyadarshi-su/naamrekha-muril_hi_n300000_e3_lr2e-05_s42_bfd254ed/tree/de5204680ed4a8ab1ac842847df6866a6e2c9251), gold F1 0.8274
  - Seed 43: [`priyadarshi-su/naamrekha-muril_hi_n300000_e3_lr2e-05_s43_223b5780`](https://huggingface.co/priyadarshi-su/naamrekha-muril_hi_n300000_e3_lr2e-05_s43_223b5780/tree/f15d615b7d760d23af6c5aa111675f0d6be39c3d), gold F1 0.8242
  - Seed 44: [`priyadarshi-su/naamrekha-muril_hi_n300000_e3_lr2e-05_s44_618532f6`](https://huggingface.co/priyadarshi-su/naamrekha-muril_hi_n300000_e3_lr2e-05_s44_618532f6/tree/aa2f28136b49859a460217190b85ebedfc625e32), gold F1 0.8174
- **Zero-shot:** score each Hindi model on the full `hi`, `mr`, and `bn` gold test sets, with no further training. Each `hi -> hi` score must be within 0.001 (about two entities) of its Phase 3 score, or the run stops before logging that seed's rows. Rows log the score measured in Phase 4, and any difference from Phase 3 is reported.
- **In-language baselines:** fine-tune `google/muril-base-cased` on 300,000 `mr` sentences and 300,000 `bn` sentences with the Phase 3 recipe, once per seed. The silver train sets have 455,248 and 961,679 sentences.
- **Training recipe:** lr 2e-5, batch size 32, 3 epochs, `max_length=128`, fp16, and warmup over the first 10% of steps.
- **Transfer gap** = in-language F1 − zero-shot F1, paired by seed, per target language and entity type. A gap whose sign changes across seeds is not treated as established.
- **Single GPU:** `CUDA_VISIBLE_DEVICES=0` is set before torch loads, and real runs stop unless exactly one GPU is visible.
- **Label alignment and scoring:** the same code as Phase 3. Words lost to truncation count as predicted `O`.
- **Precision:** every Phase 4 score comes from a freshly loaded model in full 32-bit, for both zero-shot and in-language runs. Phase 3 scored its models right after training, under fp16 autocast, so a Hindi score can differ slightly from its Phase 3 value when a near-tied prediction flips.
- **Checkpoints:** each in-language run uploads its final model to a public Hugging Face repo, confirms the upload, logs a link pinned to that revision, and only then deletes the local copy.
- **Row tagging:** every row's `notes` starts with `phase=4`, which separates it from the Phase 3 `hi -> hi` rows, and records its seed, code commit, and library versions.

## Results

_Pending._

| train \ test | hi | mr | bn |
| ------------ | -- | -- | -- |
| hi | | | |
| mr | | | — |
| bn | | — | |

## Gate

Not yet run. `uv run python scripts/phase4_gate.py` checks that the notebook ran once from a fresh kernel, and that all 15 rows (five matrix cells on three seeds) are logged once each from one code commit and match its output. It also checks the pinned Hindi checkpoints, that their scores are within 0.001 of Phase 3, the 300,000 size and single-GPU setup, and that this doc reports each cell's mean.
