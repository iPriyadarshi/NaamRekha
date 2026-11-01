# 06: Decisions Log

Material decisions live here so the team shares one history. Append a new entry whenever a choice is made that the team should not have to re-litigate: scope, ownership, the tag scheme or evaluator, model or dataset choices, and anything else binding. Newest entries at the bottom.

Format: one bullet per decision, stating what was decided and (briefly) why. Any change to the tag scheme or evaluator is a lead decision and must be recorded here (see [03-conventions.md](03-conventions.md)).

## Decisions

- **Scope and compute.** Course-project depth, nothing skipped across the four stages. Compute is free Colab + Kaggle only. Primary transformer = MuRIL (IndicBERT reported as a comparison).
- **Team and ownership.** Team is 4 people including the lead. Priyadarshi Kumar (lead) owns shared infra + the cross-lingual transfer stage; Sajan Kumar / Shreyank Sthavaramath / Ranjeet Kumar Yadav own BiLSTM-CRF / transformers / LLM+writeup respectively (roster order; reassign by strength if needed).
- **Training subset size is data-driven.** Not fixed in advance. Phase 3 grows the training set in steps and stops where the F1-vs-size curve flattens; that size becomes the working subset for everyone (see [01-roadmap.md](01-roadmap.md) Phase 3).
- **LLM prompting uses self-hosted models.** Phase 5 prompts an open HuggingFace LLM loaded and run in the same notebook (no external API), scored with the shared seqeval harness.
