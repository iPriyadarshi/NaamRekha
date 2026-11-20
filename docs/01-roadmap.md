# 01: Project Plan (phases & gates)

Work proceeds as an ordered sequence of phases. Each phase ends at a **gate**. Do not move past a gate until it is green. Update the **Status** column as you go.

| Phase | Focus                                        | Owner                                | Status      |
| ----- | -------------------------------------------- | ------------------------------------ | ----------- |
| 1     | Shared infra (loader + eval + results sheet) | Priyadarshi Kumar (Lead)             | gate passed |
| 2     | BiLSTM-CRF baseline (Hindi)                  | Sajan Kumar                          | gate passed |
| 3     | MuRIL + IndicBERT fine-tuning (Hindi)        | Shreyank Sthavaramath                | gate passed |
| 4     | Cross-lingual transfer matrix                | Priyadarshi Kumar (Lead)             | not started |
| 5     | LLM prompting + error analysis               | Ranjeet Kumar Yadav                  | not started |
| 6     | Writeup + plots                              | Ranjeet Kumar Yadav (all contribute) | not started |

Status legend: not started, in progress, gate passed, blocked

---

## Phase 1: Infrastructure (Priyadarshi Kumar / Lead)

Build the foundation everyone depends on:

- Data loader for Naamapadam (Hindi first; parametrized by language).
- `seqeval` evaluation harness (entity-level F1, per type).
- Shared **results sheet** (see [03-conventions.md](03-conventions.md) for the exact columns).

**Gate:** you can load Hindi, print the IOB2 tag distribution, and score a dummy all-`O` prediction with seqeval.

Check it with `uv run python scripts/phase1_gate.py`. It uses only `src/` and also confirms Marathi and Bengali load in the same format.

## Phase 2: BiLSTM-CRF baseline (Sajan Kumar)

fastText Hindi embeddings, then a BiLSTM, then a **CRF** layer. Start early; this is the hardest from-scratch code.

**Gate:** Hindi gold entity F1 logged in the sheet. (Rough expectation: mid-to-high 60s F1. Do **not** chase SOTA here; it's a baseline.)

Check it with `uv run python scripts/phase2_gate.py`. It confirms the BiLSTM-CRF Hindi row in the results sheet follows [03-conventions.md](03-conventions.md) and matches the output saved in the Phase 2 notebook.

**Result:** Hindi gold entity F1 **0.8038** (PER 0.869 / ORG 0.713 / LOC 0.793), trained on a 100k silver subset. This is above the expected range and becomes the bar for Phase 3. Details in [phase2/README.md](phase2/README.md).

## Phase 3: Transformer fine-tuning (Shreyank Sthavaramath)

Fine-tune MuRIL (primary) and IndicBERT (comparison) for Hindi NER, using HuggingFace `AutoModelForTokenClassification` + `Trainer`. The subword label-alignment step is the bug hotspot of this phase; budget effort for it (see [05-pitfalls.md](05-pitfalls.md)).

**Find the training size from the data; do not fix it in advance.** The full Hindi train set is roughly a million sentences, far more than a pretrained model needs, and the silver labels carry a noise ceiling, so most of that data is wasted effort. Instead, grow the training set in steps and let the gain curve tell you when to stop:

1. Fit a reusable training run that takes `subset_size` as a parameter and always evaluates on the **full** Hindi gold test set.
2. Train at an increasing sequence of sizes (for example 10k, then roughly triple each step), scoring Hindi gold F1 after each and logging it as its own row in the results sheet.
3. Plot F1 against training size as you go. Keep adding data **while the slope is steep** (each increment buys a clear F1 gain). **Stop once the curve flattens** into diminishing returns, and use that size as the working subset for the rest of the project.
4. If the curve is still climbing steeply at the largest size a free GPU session allows, report that honestly rather than forcing a plateau.

The chosen size then falls out of evidence, and the scaling curve itself is a figure for the writeup.

**Gate:** both models scored on Hindi gold; MuRIL clearly beats the BiLSTM-CRF baseline; the scaling curve is logged in the results sheet and a working subset size is chosen from it. _If MuRIL does not beat the baseline, the label alignment is broken; stop and fix before moving on._

## Phase 4: Cross-lingual transfer (Priyadarshi Kumar / Lead)

Take Shreyank Sthavaramath's best Hindi MuRIL checkpoint:

1. **Zero-shot** evaluate on Marathi and Bengali gold (no further training).
2. Fine-tune MuRIL **in-language** on `mr` and `bn` to establish the upper bound / transfer gap.

**Gate:** transfer matrix filled (train `hi`, then test `hi`/`mr`/`bn`) **plus** in-language upper bounds for `mr` and `bn`.

## Phase 5: LLM prompting + error analysis (Ranjeet Kumar Yadav)

Prompt a **self-hosted open LLM from HuggingFace**, loaded and run **in the same notebook** (no external API). Pick an instruction-tuned multilingual or Indic model that fits a free T4, for example a 1 to 3B model run directly (e.g. a Qwen2.5-Instruct small variant) or a 7 to 8B model in 4-bit via `bitsandbytes` (e.g. an Indic-tuned Llama/Gemma such as Airavata or Navarasa, or a multilingual base like Aya). Load it with `AutoModelForCausalLM` + the model's chat template; this is a **baseline**, so expect it to trail the fine-tuned MuRIL, and that gap is itself the finding.

Run both **zero-shot** and **few-shot** (a handful of in-context examples) prompting that asks the model to return PER/ORG/LOC spans in a fixed, parseable format. Then:

1. Write a **robust parser** that turns the model's free-text output back into IOB2 spans; count unparseable or malformed output as errors rather than discarding it (this is part of how an LLM baseline is judged).
2. Score with the **same** seqeval harness as every other stage.
3. Evaluate on a **sample** of the Hindi test set rather than all of it, because generation is slow on a free GPU; fix and record the sample seed and size.
4. Manage VRAM: the LLM is large, so load it after the notebook's other models are freed (`del` + `torch.cuda.empty_cache()`), or give this phase its own session.

Then Ranjeet Kumar Yadav and the lead jointly do error analysis across stages: where do the models fail (ORG spans? Bengali script tokens? LLM format drift?).

**Gate:** LLM entity-F1 (zero-shot and few-shot) in the sheet with the model name and sample size recorded; one page of concrete error examples.

## Phase 6: Writeup + plots (Ranjeet Kumar Yadav leads, all contribute)

- Comparison table (all four stages, Hindi).
- Transfer matrix rendered as a heatmap.
- The script-vs-family discussion (the research angle).

**Gate:** report done; every number in it traceable to a logged run.
