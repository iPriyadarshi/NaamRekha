# 03: Conventions (the shared contract)

**This is the most important doc.** Everything here is identical across all four notebooks. Changes are a lead decision, logged in [01-roadmap.md](01-roadmap.md).

## Tag scheme

- **IOB2**, entity types **PER / ORG / LOC**.
- Tag set: `O`, `B-PER`, `I-PER`, `B-ORG`, `I-ORG`, `B-LOC`, `I-LOC`.
- Fix a single `label2id` / `id2label` mapping in the shared loader and **import it everywhere**. Never redefine it per notebook (ordering mismatches silently corrupt results).

## Evaluation: the one rule everyone follows

- **Metric:** entity-level (span) **micro-F1** via `seqeval`. Report **per entity type** (PER/ORG/LOC) _and_ overall.
- **Never report token-level accuracy.** It looks like 95%+ because `O` dominates every sentence, and it is meaningless for NER.
- Same harness for every stage, including the LLM stage, whose text output must be parsed back into IOB2 spans before scoring.

## Results sheet: every experiment logs one row

Columns (exact):

| model | train_lang | test_lang | subset_size | epochs | F1_PER | F1_ORG | F1_LOC | F1_overall | checkpoint_path | notes |
| ----- | ---------- | --------- | ----------- | ------ | ------ | ------ | ------ | ---------- | --------------- | ----- |

Rules:

- One row per run. If it's not logged, it didn't happen.
- `subset_size` = number of training sentences actually used (we train on subsets, so be honest about it).
- `checkpoint_path` points at the saved model (Drive / Kaggle dataset) so results are reproducible.

## Data splits

- **Train on silver, evaluate on gold.** Use Naamapadam's own train/validation/test splits per language.
- **Training subset size is chosen empirically, not fixed up front.** Phase 3 grows the training set in steps and stops where the F1 curve flattens; that size becomes the working subset for everyone. Keep the subset-selection seed fixed and recorded so runs are comparable.
- **Judge the subset by entity density, not sentence count.** 100k entity-sparse sentences teach less than 100k entity-rich ones. When you build a subset, print the PER/ORG/LOC counts (the loader already reports tag distribution) and confirm it is not mostly `O`. A representative 100k beats a larger but sparse sample.
- Full gold test sets are small, so always evaluate on the **entire** gold test set, never a subset of it.

## Repo / file layout (suggested)

```
NaamRekha/
  docs/           # these guidelines
  src/            # shared loader + eval (Lead owns)
  notebooks/      # one per person per stage
  results/        # the results sheet (CSV) + saved metrics
  checkpoints/    # pointers/configs (models live on Drive/Kaggle, not in repo)
```

## Encoding

- **UTF-8 everywhere.** Devanagari and Bengali break under Windows' default console codepage. Open files with `encoding="utf-8"`, and prefer running the real work on Colab/Kaggle (Linux) rather than local PowerShell.
