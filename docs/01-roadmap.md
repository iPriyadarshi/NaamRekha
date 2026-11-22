# 01: Project Plan (phases & gates)

Work proceeds as an ordered sequence of phases. Each phase ends at a **gate**. Do not move past a gate until it is green. Update the **Status** column as you go.

| Phase | Focus                                        | Owner                                | Status      |
| ----- | -------------------------------------------- | ------------------------------------ | ----------- |
| 1     | Shared infra (loader + eval + results sheet) | Priyadarshi Kumar (Lead)             | gate passed |
| 2     | BiLSTM-CRF baseline (Hindi)                  | Sajan Kumar                          | gate passed |
| 3     | MuRIL + IndicBERT fine-tuning (Hindi)        | Shreyank Sthavaramath                | gate passed |
| 4     | Cross-lingual transfer matrix                | Priyadarshi Kumar (Lead)             | gate passed |
| 5     | LLM prompting + error analysis               | Ranjeet Kumar Yadav                  | gate passed |
| 6     | Writeup + plots                              | Ranjeet Kumar Yadav (all contribute) | in progress |

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

Check it with `uv run python scripts/phase3_gate.py`. It confirms every Phase 3 row comes from one clean notebook execution, uses the fixed training setup, links to a pinned checkpoint, and agrees with the docs.

**Result:** Hindi gold entity F1 **0.8274** (MuRIL, n=300,000, seed 42; PER 0.888 / ORG 0.744 / LOC 0.822), with a three-seed mean of 0.8230. All three seeds beat the Phase 2 baseline (0.8038). IndicBERT, untuned at the same size: 0.7839. The working subset is 300,000, chosen by a pre-set validation-F1 rule. The choice was close: 100,000 missed the cutoff by 0.0005, and gold F1 is flat from 30,000 upward. Details in [phase3/README.md](phase3/README.md).

## Phase 4: Cross-lingual transfer (Priyadarshi Kumar / Lead)

Take Shreyank Sthavaramath's best Hindi MuRIL checkpoint:

1. **Zero-shot** evaluate on Marathi and Bengali gold (no further training).
2. Fine-tune MuRIL **in-language** on `mr` and `bn` to establish the upper bound / transfer gap.

**Gate:** transfer matrix filled (train `hi`, then test `hi`/`mr`/`bn`) **plus** in-language upper bounds for `mr` and `bn`.

Check it with `uv run python scripts/phase4_gate.py`. It confirms all 15 rows (five matrix cells on three seeds) come from one clean notebook execution, use the pinned Phase 3 checkpoints and recipe, and are reported in the docs.

**Result:** across three seeds, the Hindi model scores mean zero-shot gold F1 **0.8332** on Marathi and **0.8037** on Bengali. In-language baselines score 0.8163 and 0.8132. Hindi beats the Marathi baseline on all three seeds (mean gap −0.0169), so that run is not a true upper bound, while the Bengali baseline beats Hindi on all three seeds (mean gap +0.0095). The pattern fits the shared-script idea but does not isolate script as the cause. Details in [phase4/README.md](phase4/README.md).

## Phase 5: LLM prompting + error analysis (Ranjeet Kumar Yadav)

Prompt a **self-hosted open LLM from HuggingFace**, loaded and run **in the same notebook** (no external API). Pick an instruction-tuned multilingual or Indic model that fits a free T4, for example a 1 to 3B model run directly (e.g. a Qwen2.5-Instruct small variant) or a 7 to 8B model in 4-bit via `bitsandbytes` (e.g. an Indic-tuned Llama/Gemma such as Airavata or Navarasa, or a multilingual base like Aya). Load it with `AutoModelForCausalLM` + the model's chat template; this is a **baseline**, so expect it to trail the fine-tuned MuRIL, and that gap is itself the finding.

Run both **zero-shot** and **few-shot** (a handful of in-context examples) prompting that asks the model to return PER/ORG/LOC spans in a fixed, parseable format. Then:

1. Write a **robust parser** that turns the model's free-text output back into IOB2 spans; count unparseable or malformed output as errors rather than discarding it (this is part of how an LLM baseline is judged).
2. Score with the **same** seqeval harness as every other stage.
3. Evaluate on a **sample** of the Hindi test set rather than all of it, because generation is slow on a free GPU; fix and record the sample seed and size.
4. Manage VRAM: the LLM is large, so load it after the notebook's other models are freed (`del` + `torch.cuda.empty_cache()`), or give this phase its own session.

Then Ranjeet Kumar Yadav and the lead jointly do error analysis across stages: where do the models fail (ORG spans? Bengali script tokens? LLM format drift?).

**Gate:** LLM entity-F1 (zero-shot and few-shot) in the sheet with the model name and sample size recorded; one page of concrete error examples.

Check it with `uv run python scripts/phase5_gate.py`. It confirms the parser tests, that both rows come from one clean notebook execution on all 864 gold sentences with a pinned model revision, that malformed output was counted, that the saved predictions re-score to the logged F1, and that the docs agree.

**Result:** Qwen2.5-7B-Instruct (4-bit, prompt v2) scores Hindi gold entity F1 **0.6328** zero-shot and **0.6643** few-shot (k=8), against 0.8274 for MuRIL. 11 and 5 outputs were malformed and were scored as errors (six zero-shot and all five few-shot answers were cut off at the token limit; five zero-shot answers were short with broken JSON). ORG is the weakest type (0.5956 few-shot). The cross-stage analysis finds that MuRIL's main gain over BiLSTM-CRF is recall, Marathi in-language models miss about 46 more entities than Hindi models, and tokenizer coverage does not appear to explain Bengali errors. Details in [phase5/README.md](phase5/README.md) and [phase5/error_analysis.md](phase5/error_analysis.md).

## Phase 6: Writeup + plots (Ranjeet Kumar Yadav leads, all contribute)

- Comparison table (all four stages, Hindi).
- Transfer matrix rendered as a heatmap.
- The script-vs-family discussion (the research angle).

**Gate:** report done; every number in it traceable to a logged run.

Check it with `uv run python scripts/phase6_gate.py`. It recomputes the report's numbers from `results/results.csv`, `results/error_analysis/` and the saved predictions, checks that the report states them, and checks that no placeholder is left and every link and figure exists. Figures 3 and 4 are drawn by `scripts/phase6_figures.py` from the logged results.

**Status:** a full draft is in [phase6/report.md](phase6/report.md) and the gate passes. This phase is marked passed once the team has read the report.
