# Phase 4: Cross-lingual Transfer

Everything in this folder is specific to Phase 4. The job of this phase is to measure how well a Hindi-trained MuRIL NER model transfers to Marathi (same Devanagari script) and Bengali (different script), against in-language upper bounds. All of it lives in [phase4-transfer.ipynb](../../notebooks/phase4-transfer.ipynb), which uses only the shared `src/` loader, label map, evaluator, and results logger.

## Status: ready to run

The notebook, the gate script, and the matrix/heatmap code are in place, using the final Phase 3 checkpoint, working size, and training recipe. No results are logged yet.

To finish:

1. Commit and push `phase4-cross-lingual`.
2. Run the notebook on Kaggle with `DEBUG = True`, then with `DEBUG = False`.
3. Add the results to this file, then run `uv run python scripts/phase4_gate.py`.

## Design

- **Hindi checkpoint:** [`priyadarshi-su/naamrekha-muril_hi_n300000_e3_lr2e-05_s42_bfd254ed`](https://huggingface.co/priyadarshi-su/naamrekha-muril_hi_n300000_e3_lr2e-05_s42_bfd254ed/tree/de5204680ed4a8ab1ac842847df6866a6e2c9251), pinned to revision `de5204680ed4a8ab1ac842847df6866a6e2c9251`.
- **Zero-shot:** load that checkpoint and score it on the full `hi`, `mr`, and `bn` gold test sets, with no further training. The `hi -> hi` score must reproduce the Phase 3 score of 0.8274, or the run stops before logging anything.
- **In-language upper bounds:** fine-tune `google/muril-base-cased` on 300,000 `mr` sentences and 300,000 `bn` sentences with the Phase 3 recipe. The silver train sets have 455,248 and 961,679 sentences. The only variable relative to zero-shot is the training language.
- **Training recipe:** seed 42, lr 2e-5, batch size 32, 3 epochs, `max_length=128`, fp16, and warmup over the first 10% of steps.
- **Transfer gap** = in-language F1 − zero-shot F1, per target language and per entity type.
- **Single GPU:** `CUDA_VISIBLE_DEVICES=0` is set before torch loads, and real runs stop unless exactly one GPU is visible.
- **Label alignment and scoring:** the same code as Phase 3. Words lost to truncation count as predicted `O`.
- **Checkpoints:** each in-language run uploads its final model to a public Hugging Face repo, confirms the upload, logs a link pinned to that revision, and only then deletes the local copy.
- **Row tagging:** every row's `notes` starts with `phase=4`, which separates it from the Phase 3 `hi -> hi` rows, and records the code commit and library versions.

## Results

_Pending._

| train \ test | hi | mr | bn |
| ------------ | -- | -- | -- |
| hi | | | |
| mr | | | — |
| bn | | — | |

## Gate

Not yet run. `uv run python scripts/phase4_gate.py` checks that the notebook ran once from a fresh kernel, and that the five required rows (`hi -> hi/mr/bn`, `mr -> mr`, `bn -> bn`) are logged once each from one code commit and match its output. It also checks the pinned Hindi checkpoint and score, the 300,000 size and single-GPU setup, and that this doc reports each result.
