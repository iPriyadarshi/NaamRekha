# Phase 5: Error Analysis (Hindi)

> **Status: draft.** Section 1 (MuRIL) is computed. Sections 2 and 3 (LLM) are **PENDING** the Kaggle runs. Per-sentence evidence is saved by the notebook to `results/phase5/` and the examples below are read from it.

Error kinds are entity-level, against gold: **missed** (no predicted span touches the gold entity), **boundary** (an overlapping span with different boundaries), **type confusion** (exact span, wrong type), **spurious** (a predicted span touching no gold entity), and **format drift** (LLM output that could not be parsed). Each example shows gold and prediction.

## 1. MuRIL (Phase 3 handoff, full gold test, 864 sentences)

Computed locally from the public checkpoint `priyadarshi-su/naamrekha-muril_hi_n300000_e3_lr2e-05_s42_bfd254ed` at revision `de5204680ed4a8ab1ac842847df6866a6e2c9251`. The notebook's inference code reproduces Phase 3's logged score exactly (0.8274; PER 0.8879, ORG 0.7444, LOC 0.8224). These gold-set examples were **not** used to edit the LLM prompt.

| Type | Missed | Boundary | Type confusion | Spurious | Gold entities |
| ---- | ------ | -------- | -------------- | -------- | ------------- |
| PER | 10 | 49 | 6 | 58 | 790 |
| ORG | 37 | 64 | 9 | 94 | 525 |
| LOC | 30 | 54 | 16 | 53 | 614 |
| **All** | 77 | 167 | 31 | 205 | 1,929 |

(Type = gold type, or predicted type for spurious spans. A predicted span overlapping two gold entities counts once per gold entity.)

**ORG is the weakest type, mostly through false alarms and edges, not misses.**

- **Spurious ORG (94) is the largest cell.** Predictions such as `आईपीएल`, `एनआरसी`, `इंस्टाग्राम`, `मेक इन इंडिया`, `स्वच्छ भारत मिशन`, `एमओयू`, `आरटीआई` (acronyms, schemes, brands) are tagged ORG where gold has no entity. Some of these look like real organisations, so part of this is a gold-guideline question, not only a model error.
- **Boundary ORG (64) is about titles and corporate suffixes.** Gold `वन मंत्रालय` vs predicted `मंत्रालय`; gold `केंद्र सरकार` vs `केंद्र`; gold `केएसके महानदी पावर` vs `... लिमिटेड`; gold `अखिल भारतीय कांग्रेस` vs `... पार्टी`; gold `कांग्रेस` and `विधान परिषद` (two entities) vs one merged `कांग्रेस विधान परिषद`.
- **Missed ORG (37) is short or generic institution words:** `ट्विटर` (3 times), `लोकसभा`, `संसद`, `भारत सरकार`, `निर्वाचन विभाग`, `एमएसपी`.
- **Type confusion is mainly LOC and ORG:** gold LOC predicted ORG 15 times, gold ORG predicted LOC 5 times. Examples of gold ORG predicted LOC: `सिंदरी`, `मुतंगी मस्जिद`, `नेहरू विज्ञान केंद्र`, `आस्था विद्या मंदिर`.
- **Some gold labels look odd,** so part of the ceiling is annotation, not modelling: gold ORG for `दिल्ली` and for `अन्ना`, gold ORG for `वाजपेयी`. These are reported, not corrected.

## 2. LLM zero-shot and few-shot (validation sample, then gold)

**PENDING.** To fill from the notebook's saved outputs, with gold **and** predicted span for every example:

- error counts by kind and type, for zero-shot, few-shot and MuRIL side by side;
- ORG first: false alarms (generic or organisation-sounding words), boundary errors around titles and suffixes, missed acronyms;
- sentences MuRIL gets exactly right and the LLM does not, and the reverse.

## 3. LLM format drift

**PENDING.** To fill: malformed count and reasons per mode (`empty`, `no_json`, `broken_json`, `bad_shape`), truncated outputs, answers wrapped in prose or fences, unknown labels, and the most common unaligned spans (text the model wrote that is not in the sentence, which shows translation or rewriting). Each shown with the raw output.
