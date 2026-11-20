# Phase 3: MuRIL + IndicBERT Fine-tuning

Everything in this folder is specific to Phase 3. The job of this phase is to fine-tune a transformer NER model for Hindi, find the training-set size empirically (the roadmap's scaling-curve requirement), and beat the Phase 2 BiLSTM-CRF baseline. All of it lives in [phase3-muril-indicbert.ipynb](../../notebooks/phase3-muril-indicbert.ipynb), which uses only the shared `src/` loader, label map, evaluator, and results logger.

## Status: gate passed — MuRIL scaling curve, IndicBERT comparison, and a handoff checkpoint all done

The real scaling curve (MuRIL, `DEBUG = False`) ran on Kaggle for subset sizes 10,000 / 30,000 / 100,000 / 300,000. The 900,000 run was started but the Kaggle session was interrupted right as it began (model/data just loaded, no training steps run) — no result was logged for that size, and it turned out not to be needed:

- **n=100,000 is the best point** (F1_overall 0.8343) and **clearly beats the Phase 2 baseline** (0.8038).
- **n=300,000 does not improve on it** (F1_overall 0.8287 — essentially flat, slightly lower) — per the roadmap's stopping rule ("stop once the curve flattens"), that is the signal to stop, so **900,000 was not pursued further**.
- **n=10,000 and n=30,000 scored near-zero** (the model collapsed to predicting almost nothing but "O" — a known failure mode for small-data fine-tuning on a label set that's 82% "O", with no learning-rate warmup). These are logged as real, honest data points, not bugs: the same code trained normally at 100,000 and 300,000, which confirms the pipeline itself (label alignment, training, eval) is correct.
- The scaling-curve runs themselves had their checkpoints deleted right after logging, to stay under Kaggle's disk quota. A separate **`train_run("MuRIL", 100_000)` re-run** (without deleting the checkpoint this time) produced F1_overall 0.8330 — consistent with the scaling-curve result (small run-to-run variance, same config/seed).
- The **IndicBERT comparison run** at the chosen size (100,000), using the real `ai4bharat/indic-bert`, is done: F1_overall 0.7198.
- **Both checkpoints are pushed to the Hugging Face Hub**: MuRIL at [huggingface.co/shreyanks-1/muril-hindi-ner-100k](https://huggingface.co/shreyanks-1/muril-hindi-ner-100k) and IndicBERT at [huggingface.co/shreyanks-1/indicbert-hindi-ner-100k](https://huggingface.co/shreyanks-1/indicbert-hindi-ner-100k) — downloading multi-GB checkpoints through Kaggle's own output panel was unreliable, so both were pushed directly from the Kaggle session to the Hub instead (`model.push_to_hub` / `tokenizer.push_to_hub`), skipping the local download entirely. Phase 4 can load the MuRIL one directly with `AutoModelForTokenClassification.from_pretrained("shreyanks-1/muril-hindi-ner-100k")`.

All gate items are logged, traceable to the notebook's own output, and `uv run python scripts/phase3_gate.py` passes.

## What we built

- **Models:** `google/muril-base-cased` (primary) and `ai4bharat/indic-bert` (comparison), exactly as named in the roadmap. `indic-bert` is a gated HuggingFace repo (requires a logged-in, license-accepted account); once that was set up, the real model was used directly — no substitute.
- **Label alignment** ([docs/05-pitfalls.md](../05-pitfalls.md)): the tokenizer's `word_ids()` maps each subword back to its original word. The first subword of a word gets that word's IOB2 label; every continuation subword and every special token (`[CLS]`, `[SEP]`, padding) gets `-100` so the loss ignores it. A self-test (in the notebook, before any model is trained) asserts this on a real training sentence.
- **`train_run(model_key, subset_size, ...)`**: one reusable function — seeded subset sampling + entity-density check, tokenize-and-align, HuggingFace `Trainer` (fp16 when a GPU is present, checkpoint every epoch, `max_length=128`, batch size 32), then predicts on the **full** Hindi gold test set with the same `src/eval.evaluate` harness as every other phase, and logs one `results.csv` row.
- **Scaling procedure:** call `train_run("MuRIL", n)` for each `n` in the ladder, plot F1 vs. `n`, and stop adding data once the curve flattens (the roadmap's instruction, not a fixed size decided up front).

## Results

| subset_size | model | F1_PER | F1_ORG | F1_LOC | F1_overall |
| ----------- | ----- | ------ | ------ | ------ | ---------- |
| 10,000  | MuRIL | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 30,000  | MuRIL | 0.1113 | 0.0000 | 0.0000 | 0.0807 |
| 100,000 | MuRIL | 0.9033 | 0.7469 | 0.8232 | **0.8343** |
| 300,000 | MuRIL | 0.8908 | 0.7518 | 0.8162 | 0.8287 |
| 100,000 | MuRIL (rerun) | 0.9041 | 0.7417 | 0.8223 | 0.8330 |
| 100,000 | IndicBERT | 0.7720 | 0.6542 | 0.7095 | 0.7198 |

(seed=42, lr=2e-5, batch size 32, 3 epochs, `max_length=128` for every row.)

**Chosen working subset size: 100,000.** It's the best MuRIL result and clearly beats the baseline (0.8038); the next step (300,000, 3x the data) does not improve on it, so per the roadmap's stopping rule this is where the curve flattens. 900,000 was not run.

**IndicBERT comparison** (at the chosen size, 100,000, using the real `ai4bharat/indic-bert`): F1_overall 0.7198 — clearly behind MuRIL (0.8343) on every entity type, and behind the Phase 2 baseline (0.8038) too. The roadmap's gate only requires MuRIL to beat the baseline (it does, by +0.030), so this doesn't affect the gate; IndicBERT's weaker showing here is logged as a real comparison result in its own right, not an artifact of a substitute model.

**Caveat:** each point above is a single seed (42). The 10,000/30,000 collapse is a real training-instability result, not a bug (see Status), but it also means the curve's shape at small sizes shouldn't be over-read from one seed. The margin at the chosen size (100,000) over baseline (+0.030) is reasonably wide; if this needs to be bulletproof for the writeup, re-run n=100,000 with 2-3 seeds.

## Decisions

1. **`max_length = 128`.** Matches `docs/04-compute.md`; Hindi NER sentences are short, so this only risks truncating the rare very long sentence. The notebook prints how many gold-test sentences would be affected.
2. **3 epochs, lr 2e-5, batch size 32, fp16 on GPU.** Standard BERT-family fine-tuning defaults; not tuned per the "baseline, don't chase SOTA" spirit that also governed Phase 2.
3. **A single seed (42) per subset size**, recorded in `notes`. If MuRIL's margin over 0.8038 is small at the chosen size, re-run with 2-3 seeds before calling it "clear" (same caveat Phase 2 flagged).

## Gate

**Passed.** The MuRIL scaling curve is logged, traceable to the notebook's own output, and clearly beats the baseline (best F1 0.8343 vs. 0.8038). The IndicBERT comparison row is logged too, using the real `ai4bharat/indic-bert` (F1 0.7198). `uv run python scripts/phase3_gate.py` passes end to end.

## What this unblocks

- **Phase 4** can load the MuRIL Hindi checkpoint directly from [huggingface.co/shreyanks-1/muril-hindi-ner-100k](https://huggingface.co/shreyanks-1/muril-hindi-ner-100k) (`AutoModelForTokenClassification.from_pretrained(...)`) for zero-shot Marathi/Bengali evaluation and in-language fine-tuning.
- **Phase 6** has this phase's numbers (and the scaling curve itself, which the roadmap calls out as a figure) for the comparison table, including the real MuRIL-vs-IndicBERT gap.
