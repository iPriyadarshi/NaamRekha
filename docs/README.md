# NaamRekha: Team Guidelines

Indian-language Named Entity Recognition (NER). Tag **person / location / organisation** names in Hindi and a second Indian language, with a focus on **cross-lingual zero-shot transfer** between related Indian languages.

> These docs are the single source of truth for _how we work_. They are living documents; update them as decisions are made. If a doc and a notebook disagree, fix one of them before you continue.

## How to use this folder

| Doc                                    | What it covers                                             | Primary owner         |
| -------------------------------------- | ---------------------------------------------------------- | --------------------- |
| [00-overview.md](00-overview.md)       | Problem, dataset, approach, research angle                 | Lead                  |
| [01-roadmap.md](01-roadmap.md)         | Phased plan, gates, status                                 | Lead                  |
| [02-roles.md](02-roles.md)             | Who owns what                                              | Lead                  |
| [03-conventions.md](03-conventions.md) | **The shared contract**: tag scheme, eval, results logging | Lead                  |
| [04-compute.md](04-compute.md)         | Colab / Kaggle strategy, checkpointing                     | Shreyank Sthavaramath |
| [05-pitfalls.md](05-pitfalls.md)       | Known bugs & gotchas; add as we hit them                   | Everyone              |
| [06-decisions.md](06-decisions.md)     | Decisions log: binding choices, newest at bottom           | Lead                  |

## Read this first: the three team laws

1. **One tag scheme, one evaluator.** IOB2 (PER/ORG/LOC) and `seqeval` entity-level F1, everywhere. No exceptions. See [03-conventions.md](03-conventions.md).
2. **Every number lands in the results sheet.** If an experiment isn't logged with model + train lang + test lang + subset size + F1, it didn't happen.
3. **Skip nothing, scale down instead.** We keep all four stages. When compute is tight we shrink the _data subset_, never drop a stage.

## Project at a glance

- **Scope:** course project, nothing skipped across the four stages.
- **Compute:** free Colab + Kaggle GPUs only.
- **Deliverable:** working pipeline across 4 model stages + comparison table + transfer study + report.

## Four model stages

1. BiLSTM-CRF baseline
2. MuRIL / IndicBERT fine-tuning
3. Cross-lingual zero-shot transfer (train Hindi → test Marathi / Bengali)
4. LLM prompting baseline

## Reference papers

- [1] Lample et al., _Neural Architectures for Named Entity Recognition_, NAACL 2016. https://arxiv.org/abs/1603.01360
- [2] Mhaske et al., _Naamapadam: A Large-Scale Named Entity Annotated Data for Indic Languages_, ACL 2023. https://arxiv.org/abs/2212.10168
- Dataset: https://huggingface.co/datasets/ai4bharat/naamapadam
