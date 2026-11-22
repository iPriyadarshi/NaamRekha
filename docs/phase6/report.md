# NaamRekha: Named Entity Recognition for Hindi and Cross-Lingual Transfer to Marathi and Bengali

> **Status: full draft (Phase 6).** All four stages are filled in from finished phases (1 to 5). Rule for this report: **every number comes from a row in [results.csv](../../results/results.csv) or from a saved, reproducible output, and says which one** (Appendix A maps each number to its source). Remaining before submission: a read-through by the team and the final figure layout.

## Outline

1. Introduction and research question
2. Data and evaluation method
3. Stage 1: BiLSTM-CRF baseline
4. Stage 2: MuRIL and IndicBERT fine-tuning
5. Stage 3: Cross-lingual zero-shot transfer
6. Stage 4: LLM prompting
7. Comparison across stages and error analysis
8. Script versus language family: discussion
9. Limitations, reproducibility, references
- Appendix A: where each number comes from

## 1. Introduction

We build a Named Entity Recognition system that tags person (PER), organisation (ORG) and location (LOC) spans in Hindi, then ask how far a Hindi-trained model transfers to Marathi and Bengali without further training. The four stages form a difficulty ramp: a from-scratch BiLSTM-CRF, fine-tuned multilingual transformers, zero-shot cross-lingual transfer, and prompting an open LLM. The sharpened hypothesis ([00-overview.md](../00-overview.md)): zero-shot transfer quality is gated more by **shared script** than by **language-family closeness**. Hindi to Marathi shares Devanagari and a family; Hindi to Bengali shares the family but not the script. Either outcome is a finding, provided in-language baselines are measured.

## 2. Data and evaluation method

- **Dataset:** [Naamapadam](https://huggingface.co/datasets/ai4bharat/naamapadam) (`ai4bharat/naamapadam`), 11 Indic languages; we use `hi`, `mr`, `bn`. Training labels are **silver** (projected from English NER over parallel text, so large but noisy); test labels are **gold** (human annotated). We train on silver and evaluate on gold.
- **Tags:** IOB2 over PER/ORG/LOC; fixed order `O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC` defined once in `src/data.py` and imported everywhere.
- **Hindi sizes** ([phase1/README.md](../phase1/README.md)): train 985,787, val 13,460, test 867 raw; the loader drops 3 empty test rows, leaving **864 gold test sentences** (seqeval support: PER 790, ORG 525, LOC 614 entities). Marathi and Bengali gold test sets have 1,060 and 607 sentences ([phase4/README.md](../phase4/README.md)). About 0.5% of tokens carry orphan `I-` tags from label projection; we leave them as-is for parity with the published benchmark.
- **Metric:** entity-level micro-F1 with `seqeval`, overall and per type, from the one shared `src.evaluate`. Token accuracy is never reported (it is ~95%+ because `O` dominates). The LLM's text output is parsed back into IOB2 spans first, so all four stages are scored identically.
- **Splits:** train on a seeded silver subset, select on silver validation, report on the **entire** gold test set of the target language, touched once per reported run.
- **Logging:** every run is one row in `results/results.csv` with model, languages, subset size, epochs, per-type and overall F1, checkpoint link and notes. Each phase has a gate script that checks its rows.

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
| MuRIL (mean of seeds 42, 43, 44) | 300,000 | 3 | | | | 0.8230 | Phase 3 rows for seeds 42 to 44 (0.8274, 0.8242, 0.8174) |
| IndicBERT (untuned, seed 42) | 300,000 | 1 | 0.8538 | 0.6968 | 0.7692 | 0.7839 | `results.csv` |

Findings: MuRIL beats the baseline on every seed (lowest 0.8174 vs 0.8038); the 10k, 3-epoch run collapsed to predicting no entities and recovered with 9 epochs (an under-training effect, not a data limit); gold F1 is flat from 30k upward (0.8230 to 0.8296 across size means), so the choice of 300k rests on validation, not on a clear gold gain; IndicBERT reuses MuRIL's hyperparameters on one seed, so 0.7839 is not its best possible score. **Figure:** the scaling curve is [results/phase3_scaling_curve.png](../../results/phase3_scaling_curve.png).

## 5. Stage 3: Cross-lingual zero-shot transfer

The three Phase 3 Hindi MuRIL models (seeds 42, 43, 44) are scored, with no further training, on Marathi and Bengali gold. Each target language also gets an **in-language baseline**: MuRIL fine-tuned on 300,000 sentences of that language with the Phase 3 recipe, once per seed. Every Phase 4 score is a freshly loaded model in 32-bit ([phase4/README.md](../phase4/README.md)). Mean gold F1 over three seeds (± standard deviation):

| Train \ Test | `hi` | `mr` | `bn` |
| ------------ | ---- | ---- | ---- |
| `hi` (zero-shot for `mr`, `bn`) | 0.8228 ± 0.0050 | **0.8332** ± 0.0030 | **0.8037** ± 0.0055 |
| `mr` (in-language) | | **0.8163** ± 0.0007 | |
| `bn` (in-language) | | | **0.8132** ± 0.0030 |

The `hi → hi` mean uses Phase 4's 32-bit re-score (0.8274, 0.8237, 0.8174); Phase 3 logged 0.8242 for seed 43 under fp16, hence its 0.8230 mean in Section 4. Transfer gap, paired by seed (in-language minus zero-shot):

| Target | Mean gap | Seeds 42 / 43 / 44 | Reading |
| ------ | -------- | ------------------ | ------- |
| Marathi | **−0.0169** (SD 0.0023) | −0.0142 / −0.0186 / −0.0178 | The Hindi model beats the Marathi-trained one, on all three seeds. |
| Bengali | **+0.0095** (SD 0.0051) | +0.0154 / +0.0063 / +0.0068 | In-language training helps slightly, on all three seeds. |

By type, the largest gaps are on ORG (Marathi −0.0364, Bengali +0.0165); Bengali LOC changes sign across seeds, so it is not established. **Figure:** the matrix as a heatmap is [results/phase4_transfer_matrix.png](../../results/phase4_transfer_matrix.png). Because the Hindi model beats the Marathi in-language run, that run is a baseline, not a true upper bound.

## 6. Stage 4: LLM prompting

A self-hosted open LLM (no external API) is prompted for PER/ORG/LOC spans as a JSON array, zero-shot and with 8 fixed few-shot examples, and its text output is parsed back into IOB2 by a documented, tested parser. Malformed output is counted as an error (all `O`), never dropped. The model was chosen on a 150-sentence validation sample, not on gold: `Qwen/Qwen2.5-7B-Instruct` in 4-bit (Apache-2.0, revision `a09a354...`) beat the 3B candidate by 0.11 F1, and the prompt (`v2`) was frozen before the final gold run. Setup, parser rules and validation rounds are in [phase5/README.md](../phase5/README.md). Scored on all 864 Hindi gold sentences in one clean Kaggle run:

| Setup | PER | ORG | LOC | Overall | Malformed / 864 | Source |
| ----- | --- | --- | --- | ------- | --------------- | ------ |
| LLM zero-shot | 0.6397 | 0.5644 | 0.6812 | **0.6328** | 11 | `results.csv` |
| LLM few-shot (k=8) | 0.6714 | 0.5956 | 0.7190 | **0.6643** | 5 | `results.csv` |

Examples help by 0.0315 F1 (indicative: one greedy run). Malformed outputs were 11 of 864 for zero-shot and 5 of 864 for few-shot. Six zero-shot and all five few-shot failures were answers cut off at the 320-token limit. The other five zero-shot failures were short answers with broken JSON, so a larger token limit would not fix them. The parser itself costs under 0.2 F1 points (feeding gold answers through it scores 0.9984).

## 7. Comparison across stages and error analysis

Hindi gold test, 864 sentences, entity-level micro-F1.

| Stage | Model | Train size | PER | ORG | LOC | Overall |
| ----- | ----- | ---------- | --- | --- | --- | ------- |
| 1 | BiLSTM-CRF | 100,000 | 0.8692 | 0.7130 | 0.7933 | 0.8038 |
| 2 | MuRIL (handoff) | 300,000 | 0.8879 | 0.7444 | 0.8224 | **0.8274** |
| 2 | IndicBERT | 300,000 | 0.8538 | 0.6968 | 0.7692 | 0.7839 |
| 3 | MuRIL, Hindi-trained, tested on `hi` (3-seed mean) | 300,000 | 0.8867 | 0.7323 | 0.8210 | 0.8228 |
| 4 | LLM zero-shot (Qwen2.5-7B, 4-bit) | none | 0.6397 | 0.5644 | 0.6812 | 0.6328 |
| 4 | LLM few-shot (Qwen2.5-7B, 4-bit) | none | 0.6714 | 0.5956 | 0.7190 | 0.6643 |

The ranking is MuRIL > BiLSTM-CRF > IndicBERT (untuned) > LLM few-shot > LLM zero-shot. Prompting an open 7B model is about 0.16 F1 behind the fine-tuned MuRIL, which is itself the finding: for Hindi NER with silver training data available, fine-tuning a small encoder beats prompting a larger general model.

**Error analysis** ([phase5/error_analysis.md](../phase5/error_analysis.md), 864 gold sentences, same sentences for all systems):

- **ORG is the weakest type for every model** (0.7444 MuRIL, 0.5956 LLM few-shot). Per-type, the LLM is furthest from MuRIL on PER (−0.217), because of boundaries; ORG −0.149; LOC −0.103.
- **MuRIL:** most errors are spurious spans (205 of 480) and boundaries (167); ORG alone has 94 spurious spans and 64 boundary errors.
- **LLM few-shot:** about twice MuRIL's errors (1,003). Boundary errors are over-long spans (276 of 317 longer than gold: a role or title attached to a name). False alarms are generic words (`देश`, `प्रदेश`, `राष्ट्रपति`, `पुलिस`). Misses include plain place names. Examples cut spurious spans and misses by 72 each and add 24 boundary errors.
- **Shared with MuRIL:** gold LOC predicted as ORG is the largest type confusion for both (LOC as ORG: 21 LLM, 15 MuRIL), which points to gold conventions as part of the ceiling.
- **Format drift** is small: 11 and 5 malformed outputs, plus about 30 spans per mode that do not match the sentence (altered spelling, an added word, stray characters).
- **Across all Hindi stages:** MuRIL makes the fewest errors (480), then BiLSTM-CRF (551), IndicBERT (600), LLM few-shot (1,003), and LLM zero-shot (1,122). MuRIL's gain over BiLSTM-CRF is recall (77 vs 197 missed), although it makes more spurious predictions (205 vs 148). Of 1,929 gold entities, 1,023 are right in all four compared systems and 121 are wrong in all four, including 54 ORG entities.
- **Marathi and Bengali:** across three seeds, Marathi in-language models miss about 46 more entities than Hindi models and make about 12 fewer spurious predictions. Bengali in-language models miss about 12 fewer entities than Hindi models. Both patterns have the same sign on all three seeds.

## 8. Script versus language family

**What the data show.** The Hindi model scores 0.8332 on Marathi and 0.8037 on Bengali with no target-language training. Measured against each language's in-language baseline, it does better than the baseline on Marathi (gap −0.0169, same sign on all three seeds) and slightly worse on Bengali (+0.0095, same sign on all three seeds). Hindi shares Devanagari with Marathi and a script-different family with Bengali, so the ordering of the two gaps is what the hypothesis predicts: transfer is relatively better where the script is shared.

**What the data do not show.**
- This does not prove that script is the cause. There are only two target languages, and each has a different gold test set (1,060 and 607 sentences), so absolute F1 cannot be compared across them.
- The span-level analysis shows how Marathi's in-language baseline (0.8163) falls behind: lower recall, with about 46 more missed entities than the Hindi models. It does not show why. Differences in silver-label quality are one untested possibility.
- Tokenizer coverage does not appear to explain Bengali errors. Only 0.046% of Bengali words contain `[UNK]`, and Bengali needs about as many MuRIL subwords per word as Marathi (1.305 vs 1.315).
- The seeds measure training variation, not test-set sampling variation, and they also change the sampled training subset.
- Bengali still reaches 0.8037 without a single Bengali training sentence, within about 0.01 of its in-language baseline, so a different script does not stop MuRIL's shared subword space from transferring.

**Conclusion we can defend:** the evidence is consistent with "shared script helps", but it is a two-language, single-recipe result, and it shows strong transfer to both languages. Language-family closeness alone does not explain the gap difference either, since both targets are in the same family as Hindi. We report the pattern and its limits, not a causal claim.

## 9. Limitations, reproducibility, references

- Silver training labels carry noise; the gold test sets are small (864 Hindi sentences), so differences of about a point are within seed variation (MuRIL ranges 0.8174 to 0.8274 across three seeds on Hindi).
- The BiLSTM-CRF and IndicBERT rows are single-seed and not tuned. The LLM row is one model, one frozen prompt and one greedy run, with no measured spread; the few-shot examples carry silver labels.
- An earlier 200-sentence LLM draft was run on gold test sentences and retired because it was unpinned and untraceable. Its scores are not reported, but the 3B candidate, the "copy each entity exactly as written" instruction wording, the choices of 8 few-shot examples and batch size 8, and the error-sorting code carried over from it, so the final setup may not be fully independent of the gold test set. The final model and prompt were then selected on a silver validation sample, and the gold test set was scored once with everything frozen.
- Reproducibility: pinned library versions, fixed seeds, pinned Hugging Face revisions for every checkpoint and for the LLM, and a gate script for each of Phases 1 to 5 ([scripts/](../../scripts/)) that checks the logged rows, the executed notebooks and the docs.
- References: Lample et al., *Neural Architectures for Named Entity Recognition*, NAACL 2016 ([arXiv:1603.01360](https://arxiv.org/abs/1603.01360)); Mhaske et al., *Naamapadam*, ACL 2023 ([arXiv:2212.10168](https://arxiv.org/abs/2212.10168)).

## Appendix A: where each number comes from

| Number | Source |
| ------ | ------ |
| BiLSTM-CRF 0.8038 (0.8692 / 0.7130 / 0.7933) | `results.csv`, row `BiLSTM-CRF`, Phase 2 notebook |
| MuRIL 0.8274 (0.8879 / 0.7444 / 0.8224), three-seed mean 0.8230 | `results.csv`, Phase 3 rows, `n=300000`, seeds 42 to 44 |
| IndicBERT 0.7839 | `results.csv`, row `IndicBERT` |
| Transfer matrix, gaps, per-type means | `results.csv`, rows with `phase=4` (15 rows); [phase4/README.md](../phase4/README.md) |
| `hi → hi` mean 0.8228, per-type 0.8867 / 0.7323 / 0.8210 | Phase 4 re-score rows (`in-language (Phase 3 check)`) |
| LLM 0.6328 and 0.6643, per-type, malformed 11 and 5 | `results.csv`, rows `LLM-zero-shot` and `LLM-few-shot`, commit `f1d8cbd` |
| LLM error counts, example spans, MuRIL error counts | [final_errors.csv](../../results/phase5/final_errors.csv) and [final_predictions.jsonl](../../results/phase5/final_predictions.jsonl) |
| Parser ceiling 0.9984 | `scripts/phase5_parser.py` on the Hindi gold test set (documented in [phase5/README.md](../phase5/README.md)) |
| Cross-stage and cross-language error counts, consistent wins, tokenizer and script statistics | [results/error_analysis/](../../results/error_analysis/), from [phase5-cross-stage-errors.ipynb](../../notebooks/phase5-cross-stage-errors.ipynb), commit `a53f15e` |
| Validation-sample numbers (model choice, prompt rounds) | [phase5/README.md](../phase5/README.md), from the two validation Kaggle runs (silver labels; not comparable with gold) |
