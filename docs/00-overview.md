# 00: Project Overview

## Problem statement

Build a Named Entity Recognition system that tags **PER (person)**, **LOC (location)**, and **ORG (organisation)** spans in **Hindi** text, extended to a **second Indian language** (Marathi and/or Bengali). The scientific focus is **cross-lingual zero-shot transfer** between related Indian languages.

## Dataset: Naamapadam (`ai4bharat/naamapadam`)

- Large-scale NER dataset covering **11 Indic languages**, including `hi` (Hindi), `mr` (Marathi), `bn` (Bengali), the three we care about.
- **Training data is "silver":** labels were created automatically by projecting English NER tags across parallel sentences (from the Samanantar corpus). This means the train set is **large but noisy**.
- **Test sets are "gold":** human-annotated. We **train on silver, evaluate on gold**.
- Entity types: **PER, ORG, LOC**. Tagging scheme: **IOB2** (`B-PER`, `I-PER`, `B-ORG`, `I-ORG`, `B-LOC`, `I-LOC`, `O`).

> Why the silver/gold split matters to us: there is a _noise ceiling_ on training. Chasing more silver epochs past a point won't raise gold F1. This is part of our writeup's story, not a flaw to hide.

## Approach: four stages (a difficulty ramp)

Each stage isolates one concept. We do **not** skip to the strongest model; the earlier rungs are the lessons.

1. **BiLSTM-CRF** (Lample et al. 2016): sequence labeling from scratch. Word embeddings, then a bidirectional LSTM, then a **CRF layer** that models tag-to-tag transitions. This is our baseline; it will underperform the transformers, and that is expected.
2. **MuRIL / IndicBERT fine-tuning:** transformer token classification. **MuRIL is our primary model** (pretrained on 17 Indian languages plus transliterated data; generally stronger on Indic NER). IndicBERT is reported as a comparison.
3. **Cross-lingual zero-shot transfer:** take the Hindi-trained MuRIL and evaluate it, _without further training_, on Marathi and Bengali. **This is our research contribution.**
4. **LLM prompting:** zero/few-shot prompting as a modern baseline. The interesting question: do current LLMs beat a fine-tuned encoder on low-resource Indic span extraction?

## Research angle: script vs. language family

The sharpened hypothesis (better than "transfer works"):

> Zero-shot transfer quality is gated more by **shared script** than by **language-family closeness**.

- **Hindi to Marathi:** same script (Devanagari), same family, so expect strong transfer.
- **Hindi to Bengali:** different script (Bengali), same family, so this tests whether MuRIL's shared subword space bridges the script gap.

Either outcome is a defensible finding. To make the claim, we **must** also measure in-language upper bounds (see [01-roadmap.md](01-roadmap.md), Phase 4).

## What "done" looks like

- A comparison table: all four stages on Hindi gold (entity F1, per type).
- A transfer matrix: train `hi`, then test `hi` / `mr` / `bn`, plus in-language upper bounds for `mr`/`bn`.
- An error analysis explaining _where_ transfer breaks.
- A report framing the script-vs-family result.
