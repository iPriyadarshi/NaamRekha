# notebooks/

One notebook per person per stage. Keep names predictable so results are easy to trace back.

Naming: `phase<N>-<stage>.ipynb`

Examples:

- `phase1-infra.ipynb`
- `phase2-bilstm-crf.ipynb`
- `phase3-transformers.ipynb`
- `phase4-transfer.ipynb`
- `phase5-llm-prompting.ipynb`

Conventions (see [../docs/03-conventions.md](../docs/03-conventions.md)):

- Import the shared loader and `seqeval` harness from `src/`; do not redefine the tag map or evaluator in a notebook.
- First cell pins library versions and fixes the random seed.
- Every run appends a row to `../results/results.csv`.
- Model weights go to Drive/Kaggle, never committed; only configs/pointers live under `checkpoints/`.
