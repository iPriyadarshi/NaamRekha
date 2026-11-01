# 04: Compute Strategy (free Colab + Kaggle)

We have **only free Colab and Kaggle GPUs**. The binding constraint is **session limits and disconnects**, not raw speed. Design around that.

## Three rules

1. **Kaggle for heavy fine-tuning, Colab for dev.**
   - Kaggle: longer uninterrupted sessions (~9 to 12h) and ~30 GPU-hrs/week _per account_. With 4 accounts that's plenty. Launch long MuRIL/IndicBERT runs here.
   - Colab free: disconnects aggressively (~90 min idle, variable). Use it for interactive development and debugging on a small sample.

2. **Checkpoint every epoch to persistent storage.** Assume every run dies halfway.
   - Save to Google Drive (Colab) or a Kaggle Dataset (Kaggle).
   - Set `save_strategy="epoch"` (or `save_steps`) and always be able to **resume from checkpoint**. Losing a long training run to a disconnect is the most common waste on free tiers.

3. **Subset the data, never skip a stage.** "Skip nothing" is about _stages_, not _data volume_.
   - Train on a subset of silver data, not the full million-sentence set. The working size is found empirically in Phase 3 (grow until the F1 curve flattens), and each step still fits a free session.
   - Develop/debug on **~5k** first; only scale up once the pipeline runs end-to-end.
   - Always evaluate on the **full gold test set** (it's small).

## Model sizing on a T4 (16 GB)

- Use **MuRIL-base** and **IndicBERT-base**, not the large variants. They fit comfortably.
- Batch size **16 to 32** with `fp16=True` (mixed precision). Halves memory, speeds things up.
- If OOM: lower batch size, then use gradient accumulation to keep the effective batch size.
- Cap sequence length (e.g. 128 tokens): NER sentences are short; a long `max_length` wastes memory.

## Rough run-time budget (per training run, not a schedule)

- MuRIL-base, on the order of 100k sentences, 3 epochs, T4, fp16: a few hours (figure given only to size a session, not as the chosen subset). Run on Kaggle, checkpoint each epoch.
- BiLSTM-CRF: lighter; can tolerate Colab, but still checkpoint.
- LLM prompting: a **self-hosted** HuggingFace model in the same notebook (no API). Load a 7 to 8B model in 4-bit or use a 1 to 3B model; free the other models from VRAM first. Generation is slow, so **sample** the test set rather than running all of it.

## Hygiene

- Local dev uses uv + Python 3.12 (`pyproject.toml`, `.python-version`, committed `uv.lock`). Colab/Kaggle bring their own Python and CUDA, so they do not use that env.
- Pin library versions in each notebook's first cell (HF `transformers`, `datasets`, `seqeval`, `torch`) so runs are reproducible across machines.
- Fix all random seeds and record them in the results-sheet `notes`.
