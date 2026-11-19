# Phase 2: BiLSTM-CRF Baseline

Everything in this folder is specific to Phase 2. The job of this phase was to build the from-scratch sequence-labelling baseline for Hindi NER and log its gold entity F1, so the transformer models in Phase 3 have a number to beat. All of it lives in [phase2-bilstm-crf.ipynb](../../notebooks/phase2-bilstm-crf.ipynb), which uses only the shared `src/` loader, label map, evaluator, and results logger.

## What we built

The model follows Lample et al. (2016):

- **Word features:** fastText `cc.hi.300` vectors (300-d), fine-tuned during training. Words with no fastText vector start random and are learned.
- **Character features:** a character-level BiLSTM (char dim 30, 25 hidden per direction) over each word, truncated to 20 characters. It helps with rare and unseen words.
- **Sentence encoder:** a word-level BiLSTM (128 hidden per direction) over the concatenated word and char features, with dropout 0.5 before and after.
- **Decoder:** a linear projection to the 7 tags, then a CRF (`pytorch-crf`) that picks the best tag sequence using learned transition scores.

About 30.2M parameters in total (30,207,032), most of them in the word embedding table.

Pinned libraries: `pytorch-crf==0.7.2`, `seqeval==1.2.2`; the logged run used torch 2.14.1 (CUDA 13.0) on Colab.

## Data and setup

- **Train:** a seeded random subset of 100,000 sentences from the ~985k Hindi silver train set (seed 42). Entity density check: PER 78,044 / ORG 70,041 / LOC 74,368 entities; 82.0% of tokens are `O`; 90.6% of sentences have at least one entity.
- **Model selection:** the best epoch by entity F1 on the full silver val set (13,460 sentences).
- **Report:** the full gold test set (864 sentences after the loader drops the 3 empty rows), touched once at the end. Its density: PER 788 / ORG 521 / LOC 613 entities; 83.0% of tokens are `O`; 86.0% of sentences have at least one entity, so the train subset is about as entity-rich as the test set.
- **Vocabulary:** 98,952 words, of which 77,233 (78.1%) have a fastText vector, plus 555 characters. Test-token OOV rate is 0.73%.
- **Training:** Adam (lr 1e-3), batch size 32 (3,125 steps per epoch), gradient-norm clipping at 5.0, up to 20 epochs with early stopping (patience 3 on val F1). Word dropout of 0.5 replaces singleton training words with `<unk>` so the model learns a useful unknown-word vector.
- **Compute:** Colab GPU, about 6.5 minutes per epoch. Checkpoints (`last.pt`, `best.pt`) are written every epoch to Drive, and the notebook resumes from `last.pt` automatically. The logged run resumed once, after epoch 1.

## Results

Training curve (val is silver):

| Epoch | Train loss | Val F1     |
| ----- | ---------- | ---------- |
| 1     | n/a        | 0.7424     |
| 2     | 2.973      | 0.7484     |
| 3     | 2.566      | 0.7517     |
| 4     | 2.312      | **0.7546** |
| 5     | 2.104      | 0.7480     |
| 6     | 1.937      | 0.7402     |
| 7     | 1.803      | 0.7447     |

Epoch 1 ran in an earlier session; only its val F1 survives (from the resume message). Val F1 peaked at epoch 4 and early stopping ended the run after epoch 7.

Best checkpoint (epoch 4) on Hindi **gold** test:

| Type        | Precision | Recall | F1         | Support |
| ----------- | --------- | ------ | ---------- | ------- |
| PER         | 0.8638    | 0.8747 | 0.8692     | 790     |
| ORG         | 0.7448    | 0.6838 | 0.7130     | 525     |
| LOC         | 0.8123    | 0.7752 | 0.7933     | 614     |
| **Overall** | 0.8169    | 0.7911 | **0.8038** | 1929    |

Overall is seqeval micro-average. This is the row logged in [results.csv](../../results/results.csv).

## Findings

- **The baseline is stronger than planned.** The roadmap expected mid-to-high 60s; we got 0.804. Two likely reasons: the model trains on noisy silver labels but is scored on cleaner gold labels (gold test F1 is about 5 points above silver val F1), and the vocabulary covers test words through fastText (see decision 1 below), which brings test-token OOV down to 0.73%.
- **ORG is the weakest type** (F1 0.713, recall 0.684). PER is the strongest (0.869). ORG errors are the first place to look in Phase 5's error analysis.
- **Precision is above recall for every type.** The model misses entities more often than it invents them.
- **Gold support matches the Phase 1 contract.** Seqeval's support counts are 7 higher than the `B-` tag counts (788 / 521 / 613), which equals the 7 orphan `I-` tags Phase 1 found in gold test. Seqeval counts each orphan as an entity, as recorded in [../phase1/README.md](../phase1/README.md).
- **The model overfits after a few epochs.** Train loss keeps falling while val F1 peaks at epoch 4, so early stopping on val is doing real work.

## Decisions

1. **Vocabulary = train-subset words + val/test words that have a fastText vector.** Pretrained vectors are unsupervised, so this leaks no labels and follows Lample et al. Test-only words keep their raw fastText vectors, since they never receive gradient updates. This must be stated wherever the 0.804 is reported.
2. **The CRF is unconstrained.** It does not forbid `O -> I-X`, because the silver data keeps its ~0.5% orphan `I-` tags (Phase 1, decision 2) and the CRF should learn the transitions from the data.
3. **Training subset fixed at 100k for this baseline.** Phase 3 picks the project-wide working subset from its scaling curve; the BiLSTM-CRF is not re-run on that size unless the writeup needs a like-for-like comparison.
4. **A single seed (42).** Enough for a baseline. If MuRIL's margin over 0.804 is small, re-run with 2 to 3 seeds before calling it "clear".

## Artifacts

- **Notebook:** [phase2-bilstm-crf.ipynb](../../notebooks/phase2-bilstm-crf.ipynb), committed with its outputs.
- **Checkpoint folder** on shared Drive, linked from `checkpoint_path` in [results.csv](../../results/results.csv): `NaamRekha/phase2/bilstm-crf_hi_n100000_s42/`.
  - `best.pt`: epoch-4 weights plus `itos`, `chars`, labels, and val scores. Use this one for evaluation.
  - `last.pt`: latest epoch with optimizer state and history. Only needed to resume training.
- **fastText cache:** `NaamRekha/fasttext/cc.hi.300.filtered_bilstm-crf_hi_n100000_s42.pt`. Only the vectors this vocabulary needs, so reruns skip the 1.2 GB download. Not a model.

## Gate

**Passed.** Hindi gold entity F1 (0.8038) is logged in the results sheet with its checkpoint and run settings.

## What this unblocks

- **Phase 3 has its bar.** The gate requires MuRIL to clearly beat **0.804** Hindi gold F1, not the high-60s originally expected. If it does not, check the subword label alignment first ([../05-pitfalls.md](../05-pitfalls.md)).
- **A reusable checkpoint.** `best.pt` (weights, `itos`, `chars`, labels) is in the shared Drive folder linked from the results sheet, for error analysis in Phase 5 and the comparison table in Phase 6.
