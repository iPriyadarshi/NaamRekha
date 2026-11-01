# 02: Team Roles

Team of 4, and the lead is one of the four (not separate). We split by **model stage, on top of shared infrastructure**, not by language. This keeps one person responsible for the loader + evaluator so all notebooks produce comparable numbers.

## Roster

| Name                  | GitHub        | Role on the project |
| --------------------- | ------------- | ------------------- |
| Priyadarshi Kumar     | [@iPriyadarshi](https://github.com/iPriyadarshi) | Lead   |
| Sajan Kumar           | [@Sajan0304](https://github.com/Sajan0304)       | Member |
| Shreyank Sthavaramath | [@shreyanks-1](https://github.com/shreyanks-1)   | Member |
| Ranjeet Kumar Yadav   | [@Ranjeet2702](https://github.com/Ranjeet2702)   | Member |

> Stage assignments below follow the roster order as a starting point. Reassign by strength if it makes sense; record any change in the decisions log ([06-decisions.md](06-decisions.md)).

## Ownership

| Owner                        | Owns                                                                                                       | Deliverable                                                      |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| **Priyadarshi Kumar (Lead)** | Shared data loader, `seqeval` harness, results sheet, integration; cross-lingual transfer matrix (Phase 4) | Comparable numbers for everyone + the research-angle experiments |
| **Sajan Kumar**              | BiLSTM-CRF baseline (Hindi)                                                                                | Baseline F1 logged; from-scratch sequence model                  |
| **Shreyank Sthavaramath**    | MuRIL + IndicBERT fine-tuning (Hindi); owns [04-compute.md](04-compute.md)                                 | Headline transformer results                                     |
| **Ranjeet Kumar Yadav**      | LLM prompting baseline + report/plots                                                                      | Modern baseline + final writeup                                  |

## Why this split

- **Shared infra is owned by one person (the lead).** The classic student-team failure is four people each writing their own evaluator, then being unable to compare numbers. One loader, one metric, one sheet.
- **Stages are sequential dependencies.** The transfer stage needs Shreyank Sthavaramath's checkpoint; Ranjeet Kumar Yadav's writeup needs everyone's numbers; everything needs the harness from Phase 1. The [plan](01-roadmap.md) sequences these so no one is blocked waiting.
- **Everyone adds to [05-pitfalls.md](05-pitfalls.md).** Gotchas found by one person save the others hours.

## Working agreements

- Branch/notebook per person; don't edit someone else's stage notebook without asking.
- Any change to the tag scheme or evaluator is a **lead decision** and goes in the decisions log ([06-decisions.md](06-decisions.md)).
- Sync at each gate: demo the number, log it, unblock the next person.
