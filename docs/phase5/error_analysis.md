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

## 2. LLM zero-shot and few-shot

**2a. Validation sample (preliminary).** Qwen2.5-7B-Instruct (4-bit), prompt v2, 150 Hindi **val** sentences with **silver** labels, with MuRIL on the same sentences. These counts guide the analysis; the report uses the final gold-run counts (**PENDING**).

| System | Boundary | Missed | Spurious | Type confusion | Total |
| ------ | -------- | ------ | -------- | -------------- | ----- |
| LLM zero-shot | 62 | 70 | 98 | 8 | 238 |
| LLM few-shot | 87 | 43 | 62 | 15 | 207 |
| MuRIL | 42 | 17 | 33 | 9 | 101 |

ORG per kind (boundary / missed / spurious / type confusion): few-shot 39 / 27 / 25 / 10, zero-shot 20 / 45 / 32 / 4, MuRIL 22 / 9 / 15 / 6. PER few-shot: 28 / 8 / 16 / 2. LOC few-shot: 20 / 8 / 21 / 3.

- **Examples change the kind of error.** Few-shot cuts spurious spans (98 to 62) and misses (70 to 43), but boundary errors rise (62 to 87): the model finds more real entities and places their edges less well.
- **The LLM makes about twice MuRIL's errors** in every kind except type confusion.
- **ORG is about half of all errors** for both systems (few-shot 101 of 207; MuRIL 52 of 101). Examples cut ORG misses (45 to 27) and double ORG boundary errors (20 to 39).

**2b. Concrete examples (same validation run; LLM few-shot, MuRIL where noted).** Gold is silver here, so each example is marked as a likely **model error** or a likely **label quirk**. They come from the first rows of the notebook's error tables, not a random sample; the gold-run examples replace them.

| Kind | Gold | LLM predicted | Reading |
| ---- | ---- | ------------- | ------- |
| spurious | none | ORG `बिजली कंपनियों`, ORG `रिपोर्ट`, ORG `सीएनजी` | **Model error:** generic nouns and a fuel tagged as organisations. |
| spurious | none | PER `राष्ट्रपति`, PER `सलामी` | **Model error:** a title and the word "opening (batsman)" tagged as persons. |
| type confusion | ORG `बसपा`, ORG `तेदेपा`, ORG `लोकसभा` | LOC for each | **Model error:** party acronyms and the Lok Sabha tagged as places (also acronyms the model misses entirely). |
| boundary | ORG `रेलवे भर्ती बोर्ड` and ORG `RRB` (two entities) | one ORG `रेलवे भर्ती बोर्ड ( RRB )` | Merges name and acronym; gold splits them. |
| boundary | ORG `वैश्विक मंच` and ORG `सचिवालय` | one ORG `वैश्विक मंच के सचिवालय` | Merges two entities into one span (MuRIL makes the same kind of merge: `सीपीआई एमएल लिबरेशन` vs gold `सीपीआई` and `एमएल`). |
| boundary | PER `रामनाथ कोविंद` | PER `राष्ट्रपति रामनाथ कोविंद` | Title included in the name span. |
| missed | ORG `संस्थान`, ORG `राष्ट्रीय महत्व के संस्थान` | nothing | Generic institution phrases; also missed LOC `मसूर`, `अंगलियात`, `राज्यों`, PER `पद्मावती`. |
| type confusion | ORG `तिरुवनंतपुरम`, ORG `मध्यप्रदेश` | LOC for each | **Label quirk:** gold calls cities and states ORG; MuRIL predicts LOC for the same two. |
| boundary / missed | ORG spans running through a whole clause, e.g. `मानव संसाधन विकास मंत्रालय ने छात्रों , शिक्षक...` | `मानव संसाधन विकास मंत्रालय` | **Label quirk:** the gold span is silver noise; both the LLM and MuRIL predict the sensible short span. |
| spurious / type | LOC `व्हाइट हाउस` | ORG `व्हाइट हाउस` | Arguable: the model's reading (an institution) is defensible. |

Takeaways: (1) the LLM's real ORG failures are false alarms on generic words, acronyms read as places, and merged or over-long spans; (2) a visible share of the "errors" on both systems come from silver gold labels, so the gold test set must be used for the report numbers; (3) MuRIL fails on the same noisy labels, which suggests those are annotation noise, not LLM weakness.
- Still to fill from the final gold run: the same table on 864 gold sentences, sentences MuRIL gets exactly right and the LLM does not (and the reverse).

## 3. LLM format drift

**Validation sample (Qwen2.5-7B, prompt v2, 150 sentences).** Format drift is not the problem for this model with this prompt: 0 malformed outputs in both modes, no answers wrapped in prose or fences, no invalid items, no unknown labels. One zero-shot answer hit the 320-token limit yet still parsed. The drift that does exist is **the model not copying the sentence exactly** (unaligned spans: 6 zero-shot, 8 few-shot; overlap-dropped 1 and 3). The unaligned spans the model wrote that are not in the sentence:

- **Altered spelling or dropped letters:** `सप्रीम कोर्ट` (for `सुप्रीम कोर्ट`), `अंगियात` (gold `अंगलियात`), `मसू` (gold `मसूर`), `सिद्धारथ दोहरी` (twice), `भारतीय बास्केट...` (cut short).
- **Names rewritten into Latin script:** `bihar`, `chapra`, `saran`. The instruction said "same script, no translation", and few-shot still did this three times.
- **Wrong or invented entities:** `राहुल जिन्ना`, `न्यॉर्क`, and a long run-on `मज़दूर - किसान - छोटा व्यापारी भारत`.

These count as lost entities (the span matches nothing), so they hurt recall without being counted as malformed. The final gold run will report malformed, truncated, noisy, invalid, unknown-label and unaligned counts per mode from `results/phase5/`.

**MuRIL versus LLM, whole-sentence view (150 validation sentences).** Sentences with every entity exactly right:

| | Both exact | MuRIL only | LLM only | Neither |
| --- | --- | --- | --- | --- |
| LLM zero-shot | 38 | 47 | 7 | 58 |
| LLM few-shot | 42 | 43 | 7 | 58 |

MuRIL is exactly right on a sentence where the LLM is not about six times as often as the reverse (43 vs 7 for few-shot). Gold-run version: **PENDING**.
