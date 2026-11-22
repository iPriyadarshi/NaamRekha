# 05: Pitfalls & Gotchas

Known traps. **Add to this as the team hits new ones**: a gotcha one person solves saves the others hours.

## The big one: subword label alignment (transformers)

BERT-family tokenizers split a word into multiple wordpieces, but we have **one label per word**. Misaligning here is the #1 NER bug.

- Label only the **first subword** of each original token with its tag. If a SentencePiece tokenizer emits a standalone `▁` first, label the first real piece after it.
- Set every **continuation subword** label to `-100` (so the loss ignores it).
- Also `-100` the special tokens (`[CLS]`, `[SEP]`, padding).
- Use the tokenizer's `word_ids()` to map subwords back to words.

**Symptom it's broken:** MuRIL scores _worse than or equal to_ the BiLSTM-CRF, or F1 is weirdly low with high token accuracy. If the transformer gate (Phase 3) fails, check this first.

## Evaluation traps

- **Token accuracy looks amazing and means nothing**: `O` is ~90% of tokens. Only entity-level seqeval F1 counts.
- **IOB vs IOB2 mismatch** between prediction and gold will tank F1 silently. One scheme, fixed in the loader.
- **Label-id ordering** must be identical across notebooks. Import the shared `label2id`; never redefine it.

## Cross-lingual transfer traps

- **No upper bound = no story.** "Zero-shot got 65 F1" is meaningless without the in-language ceiling. Always run the in-language fine-tune too.
- Expect **Hindi to Marathi (same Devanagari script) to transfer better than Hindi to Bengali (different script)**, but _measure_ it; that gap is the finding.

## Data traps

- Training data is **silver/noisy**: more epochs on it won't keep raising gold F1. Watch for the plateau.
- `O` dominates, so there is class imbalance. Entity-level F1 handles this correctly; don't "fix" it by resampling unless you measure the effect.

## Compute traps

- **Disconnects lose unsaved runs.** Checkpoint every epoch to Drive/Kaggle (see [04-compute.md](04-compute.md)).
- **OOM on T4:** lower batch size + gradient accumulation + `fp16=True` + shorter `max_length`.

## Windows / encoding traps

- **UTF-8 everywhere.** Devanagari/Bengali render as `???` or mojibake under PowerShell's default codepage. Do the real work on Colab/Kaggle (Linux); if reading locally, pass `encoding="utf-8"` explicitly.

---

### Add new gotchas below (name + what fixed it)

- **Kaggle `DataParallel` doubles the batch size.** On T4 x2, `Trainer` silently splits each batch across both GPUs, so `batch_size=32` runs as 64 with half the steps. Set `CUDA_VISIBLE_DEVICES=0` before importing torch and check that the GPU count is 1.
- **All-`O` collapse can be under-training.** MuRIL at 10k with lr 2e-5 and 3 epochs (939 steps) predicted no entities, but recovered to gold F1 0.8231 with 9 epochs. Check the step count before blaming data size.
- **Standalone `▁` in IndicBERT.** About 2.9% of sampled Hindi words start with a bare `▁` piece. Labeling it puts the tag on a piece with no text. Use the notebook's `label_positions` helper.
- **Kaggle has its own `/kaggle/src`.** Checking for `../src` makes a notebook think `/kaggle` is the repo. Check for a real repo file such as `src/data.py`.
- **Upload before deleting checkpoints.** Clearing Kaggle disk before saving a model elsewhere makes its result impossible to reload or verify. Upload, confirm, log the pinned link, then delete.
- **An in-language run is not automatically an upper bound.** In Phase 4, Hindi models beat Marathi-trained models on Marathi gold on all three seeds (mean 0.8332 vs 0.8163). Check the result across seeds before interpreting a small transfer gap.
- **The fp16 training object scores differently from a reloaded model.** `Trainer` leaves an autocast wrapper on `model.forward`, so scoring it right after training uses fp16, while a reloaded model uses 32-bit. A near-tied prediction can flip. Save, reload, then score when comparing models.
- **LLM spans do not match Naamapadam tokens.** The dataset splits punctuation and initials into their own tokens (`एम .`), while a model copies `एम.`. Matching exact word tokens drops correct entities. Align on text with whitespace and zero-width joiners removed, after Unicode NFC (`scripts/phase5_parser.py`). Feeding gold answers through that parser scores 0.9984 F1 on Hindi gold, so the parser itself costs under 0.2 points.
- **Check an LLM's license and gating before choosing it.** `Qwen2.5-3B-Instruct` is under the research-only `qwen-research` license, while the 7B is Apache-2.0. Airavata and Aya Expanse 8B are gated. Read both from the Hub, and pin the exact revision.
- **`seqeval` has no `__version__`.** Reading it raises `AttributeError`, which can stop a notebook after the model has loaded. Use `importlib.metadata.version("seqeval")` to record library versions.
