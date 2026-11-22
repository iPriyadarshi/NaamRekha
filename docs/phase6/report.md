# NaamRekha: Named Entity Recognition for Hindi and Cross-Lingual Transfer to Marathi and Bengali

> **Status: working draft (Phase 6).** Sections 1 to 4 are drafted from finished phases (1 to 3). Sections 5 to 8 wait on Phase 4 (transfer) and the final Phase 5 gold run, and are marked **PENDING**. Rule for this report: **every number comes from a row in [results.csv](../../results/results.csv) or from a saved, reproducible output, and says which one.** Phase 4 numbers are not used until Phase 4 is final.

## Outline

1. Introduction and research question
2. Data and evaluation method
3. Stage 1: BiLSTM-CRF baseline
4. Stage 2: MuRIL and IndicBERT fine-tuning
5. Stage 3: Cross-lingual zero-shot transfer **(PENDING Phase 4)**
6. Stage 4: LLM prompting **(PENDING final Phase 5 run)**
7. Comparison across stages and error analysis
8. Script versus language family: discussion **(PENDING Phase 4)**
9. Limitations, reproducibility, references

## 1. Introduction

We build a Named Entity Recognition system that tags person (PER), organisation (ORG) and location (LOC) spans in Hindi, then ask how far a Hindi-trained model transfers to Marathi and Bengali without further training. The four stages form a difficulty ramp: a from-scratch BiLSTM-CRF, fine-tuned multilingual transformers, zero-shot cross-lingual transfer, and prompting an open LLM. The sharpened hypothesis ([00-overview.md](../00-overview.md)): zero-shot transfer quality is gated more by **shared script** than by **language-family closeness**. Hindi to Marathi shares Devanagari and a family; Hindi to Bengali shares the family but not the script. Either outcome is a finding, provided in-language upper bounds are measured.

## 2. Data and evaluation method

- **Dataset:** [Naamapadam](https://huggingface.co/datasets/ai4bharat/naamapadam) (`ai4bharat/naamapadam`), 11 Indic languages; we use `hi`, `mr`, `bn`. Training labels are **silver** (projected from English NER over parallel text, so large but noisy); test labels are **gold** (human annotated). We train on silver and evaluate on gold.
- **Tags:** IOB2 over PER/ORG/LOC; fixed order `O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC` defined once in `src/data.py` and imported everywhere.
- **Hindi sizes** ([phase1/README.md](../phase1/README.md)): train 985,787, val 13,460, test 867 raw; the loader drops 3 empty test rows, leaving **864 gold test sentences** (seqeval support: PER 790, ORG 525, LOC 614 entities). About 0.5% of tokens carry orphan `I-` tags from label projection; we leave them as-is for parity with the published benchmark.
- **Metric:** entity-level micro-F1 with `seqeval`, overall and per type, from the one shared `src.evaluate`. Token accuracy is never reported (it is ~95%+ because `O` dominates). The LLM's text output is parsed back into IOB2 spans first, so all four stages are scored identically.
- **Splits:** train on a seeded silver subset, select on silver validation, report on the **entire** gold test set (864 sentences), touched once per reported run.
- **Logging:** every run is one row in `results/results.csv` with model, languages, subset size, epochs, per-type and overall F1, checkpoint link and notes.

## 3. Stage 1: BiLSTM-CRF

Lample et al. (2016): fastText `cc.hi.300` word vectors plus a character BiLSTM, a word-level BiLSTM (128 hidden per direction), and a CRF decoder (about 30.2M parameters). Trained on a seeded 100,000-sentence silver subset (seed 42), best epoch by silver validation F1 (epoch 4 of 7). Details and caveats (including how the vocabulary uses fastText vectors for test words) are in [phase2/README.md](../phase2/README.md).

| Model | Train size | PER | ORG | LOC | Overall | Source |
| ----- | ---------- | --- | --- | --- | ------- | ------ |
| BiLSTM-CRF | 100,000 | 0.8692 | 0.7130 | 0.7933 | **0.8038** | `results.csv`, seed 42 |

The roadmap expected the mid-to-high 60s; 0.8038 is well above that. Likely reasons given in Phase 2: noisy silver training but cleaner gold scoring, and fastText coverage of test words (test-token OOV 0.73%). Single seed.

## 4. Stage 2: MuRIL and IndicBERT

`google/muril-base-cased` (primary) and `ai4bharat/indic-bert` (comparison) with `AutoModelForTokenClassification`. Labels sit on the first subword of each word; continuation subwords and special tokens get `-100`. The training size was found from the data: a scaling ladder at 10k, 30k, 100k and 300k sentences, with a pre-set rule on **validation** F1 (not gold) selecting **300,000**. All numbers are in [phase3/README.md](../phase3/README.md).

| Model | Train size | Seeds | PER | ORG | LOC | Overall | Source |
| ----- | ---------- | ----- | --- | --- | --- | ------- | ------ |
| MuRIL (handoff, seed 42) | 300,000 | 1 | 0.8879 | 0.7444 | 0.8224 | **0.8274** | `results.csv` |
| MuRIL (mean of seeds 42, 43, 44) | 300,000 | 3 | | | | 0.8230 | `results.csv` rows for seeds 42 to 44 (0.8274, 0.8242, 0.8174) |
| IndicBERT (untuned, seed 42) | 300,000 | 1 | 0.8538 | 0.6968 | 0.7692 | 0.7839 | `results.csv` |

Findings to carry into the report: MuRIL beats the baseline on every seed (lowest 0.8174 vs 0.8038); the 10k, 3-epoch run collapsed to predicting no entities and recovered with 9 epochs (an under-training effect, not a data limit); gold F1 is flat from 30k upward (0.8230 to 0.8296 across size means), so the choice of 300k rests on validation, not on a clear gold gain; IndicBERT reuses MuRIL's hyperparameters on one seed, so 0.7839 is not its best possible score. **Figure:** the scaling curve is [results/phase3_scaling_curve.png](../../results/phase3_scaling_curve.png).

## 5. Stage 3: Cross-lingual zero-shot transfer

**PENDING Phase 4.** The Hindi-trained MuRIL (the handoff model above) is evaluated without further training on Marathi and Bengali gold, and compared with in-language fine-tuned upper bounds. Fill from the final three-seed Phase 4 rows of `results.csv`; do not use partial results.

| Train \ Test | `hi` | `mr` | `bn` |
| ------------ | ---- | ---- | ---- |
| `hi` (zero-shot for `mr`, `bn`) | 0.8274 (seed 42, Stage 2 row) | PENDING | PENDING |
| in-language upper bound | n/a | PENDING | PENDING |

Plan for the figure: this matrix as a heatmap (rows = training language, columns = test language), annotated with F1 and the transfer gap to the in-language ceiling.

## 6. Stage 4: LLM prompting

**PENDING final Phase 5 gold run.** Setup, parser rules and the draft procedure are in [phase5/README.md](../phase5/README.md). Self-hosted open LLM (no external API), zero-shot and few-shot, scored on all 864 gold sentences with the shared evaluator; malformed output counts as an error (all `O`) and is reported.

| Setup | PER | ORG | LOC | Overall | Malformed / 864 | Source |
| ----- | --- | --- | --- | ------- | --------------- | ------ |
| LLM zero-shot | PENDING | PENDING | PENDING | PENDING | PENDING | `results.csv` |
| LLM few-shot | PENDING | PENDING | PENDING | PENDING | PENDING | `results.csv` |

## 7. Comparison across stages and error analysis

Hindi gold test, 864 sentences, entity-level micro-F1.

| Stage | Model | Train size | PER | ORG | LOC | Overall |
| ----- | ----- | ---------- | --- | --- | --- | ------- |
| 1 | BiLSTM-CRF | 100,000 | 0.8692 | 0.7130 | 0.7933 | 0.8038 |
| 2 | MuRIL (handoff) | 300,000 | 0.8879 | 0.7444 | 0.8224 | 0.8274 |
| 2 | IndicBERT | 300,000 | 0.8538 | 0.6968 | 0.7692 | 0.7839 |
| 3 | MuRIL, Hindi-trained, tested on `hi` | 300,000 | see Stage 2 | | | 0.8274 |
| 4 | LLM zero-shot | none | PENDING | PENDING | PENDING | PENDING |
| 4 | LLM few-shot | none | PENDING | PENDING | PENDING | PENDING |

Error analysis: [phase5/error_analysis.md](../phase5/error_analysis.md). So far (MuRIL, computed from its public checkpoint): ORG is the weakest type for every model, with most MuRIL ORG errors being spurious spans (94) and boundary errors (64) rather than misses (37); gold LOC predicted ORG is the largest type confusion (15). LLM error comparison and Phase 4 cross-language errors are PENDING.

## 8. Script versus language family

**PENDING Phase 4.** State the measured result for Hindi to Marathi (shared script) versus Hindi to Bengali (different script), relative to each language's in-language ceiling, and say whether the data supports "script gates transfer more than family". Either outcome is reportable; do not write the conclusion before the numbers.

## 9. Limitations, reproducibility, references

- Silver training labels carry noise; gold test has only 864 Hindi sentences, so differences of about a point are within seed variation (MuRIL ranges 0.8174 to 0.8274 across three seeds).
- The BiLSTM-CRF and IndicBERT rows are single-seed and not tuned.
- The LLM stage uses one model, one frozen prompt and greedy decoding.
- Reproducibility: pinned library versions, fixed seeds, pinned Hugging Face revisions for every checkpoint, and gate scripts for Phases 1 to 5 ([scripts/](../../scripts/)).
- References: Lample et al., *Neural Architectures for Named Entity Recognition*, NAACL 2016 ([arXiv:1603.01360](https://arxiv.org/abs/1603.01360)); Mhaske et al., *Naamapadam*, ACL 2023 ([arXiv:2212.10168](https://arxiv.org/abs/2212.10168)).
