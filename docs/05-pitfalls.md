# 05: Pitfalls & Gotchas

Known traps. **Add to this as the team hits new ones**: a gotcha one person solves saves the others hours.

## The big one: subword label alignment (transformers)

BERT-family tokenizers split a word into multiple wordpieces, but we have **one label per word**. Misaligning here is the #1 NER bug.

- Label only the **first subword** of each original token with its tag.
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

- _(none yet)_
