# Phase 5: Error Analysis

Sections 1 to 4 cover the LLM and MuRIL on Hindi gold. Sections 5 to 7 compare all stages and the Marathi and Bengali transfer models.

Counts in Sections 1 to 4 come from the final Kaggle run (commit `f1d8cbd`): [final_errors.csv](../../results/phase5/final_errors.csv) and [final_predictions.jsonl](../../results/phase5/final_predictions.jsonl), the same sentences for all three systems. LLM = `Qwen/Qwen2.5-7B-Instruct` (4-bit), prompt `v2`, scored in [README.md](README.md). MuRIL = the Phase 3 handoff checkpoint, re-scored in the same notebook at 0.8274.

Error kinds are entity-level, against gold: **missed** (no predicted span touches the gold entity), **boundary** (an overlapping span with different edges), **type confusion** (exact span, wrong type), **spurious** (a predicted span touching no gold entity), and **format drift** (LLM output that could not be parsed). A prediction overlapping two gold entities counts once per gold entity. Type = gold type, or the predicted type for spurious spans.

## 1. Error counts

| System | Boundary | Missed | Spurious | Type confusion | Total |
| ------ | -------- | ------ | -------- | -------------- | ----- |
| LLM zero-shot | 293 | 299 | 478 | 52 | 1,122 |
| LLM few-shot | 317 | 227 | 406 | 53 | 1,003 |
| MuRIL | 167 | 77 | 205 | 31 | 480 |

By type (boundary / missed / spurious / type confusion):

| System | PER | ORG | LOC |
| ------ | --- | --- | --- |
| LLM few-shot | 148 / 107 / 107 / 10 | 86 / 71 / 178 / 19 | 83 / 49 / 121 / 24 |
| MuRIL | 49 / 10 / 58 / 6 | 64 / 37 / 94 / 9 | 54 / 30 / 53 / 16 |

- **The LLM makes about twice MuRIL's errors** (1,003 vs 480), and three times as many misses (227 vs 77). On whole sentences with every entity exactly right: both 275, MuRIL only 254, LLM only 67, neither 268 (few-shot; zero-shot 257 / 272 / 66 / 269).
- **Examples change the mix.** Few-shot removes 72 spurious spans and 72 misses against zero-shot, at the cost of 24 more boundary errors. Type confusion is unchanged (52 vs 53).
- **Spurious spans are the largest kind for both systems,** and ORG has the most (178). PER has the most boundary (148) and missed (107) errors.
- **The gap to MuRIL differs by type:** PER −0.217, ORG −0.149, LOC −0.103 F1 (few-shot). The LLM is furthest from MuRIL on PER, because of boundaries (below), not on ORG.

## 2. What the LLM gets wrong (few-shot, gold test)

| Kind | Gold | LLM predicted |
| ---- | ---- | ------------- |
| boundary (PER) | `अमित शाह`, `राजनाथ सिंह`, `अक्षर पटेल`, `अनुराग बासु` | `गृह मंत्री अमित शाह`, `रक्षा मंत्री श्री राजनाथ सिंह`, `स्पिनर अक्षर पटेल`, `श्रीदेवी डायरेक्टर अनुराग बासु` |
| boundary (PER) | `ज्योतिरादित्य एम . सिंधिया`, `मनोज चुघ` | `श्री ज्योतिरादित्य एम . सिंधिया`, `मनोज चुघ धरना` |
| boundary (ORG) | `कांग्रेस`, `पाकिस्तान तहरीक - ए - इंसाफ` | `कांग्रेस पार्टी`, `पाकिस्तान तहरीक - ए - इंसाफ पार्टी` |
| boundary (ORG) | `केन्द्र सरकार`, `आय समिति` | `सरकार`, `समिति` (too short) |
| boundary (ORG) | `ऑटोमोटिव टायर मैन्युफैक्चरर्स एसोसिएशन` and `एटमा` (two entities) | one span `ऑटोमोटिव टायर मैन्युफैक्चरर्स एसोसिएशन ( एटमा )` |
| spurious | none | LOC `देश` (7), LOC `प्रदेश` (7), PER `राष्ट्रपति` (4), PER `राज्यपाल` (4), PER `प्रधानमंत्री` (3) |
| spurious | none | ORG `पुलिस` (4), `मंत्रालय` (3), `विधानसभा` (3), `विश्वविद्यालय` (3), `मेक इन इंडिया` (2), `स्वच्छ भारत मिशन` (2) |
| missed | LOC `भारत` (5), `ईरान` (3), `अमेरिका` (2), `हप्र` (4); PER `सलमान खान` (2) | nothing |
| missed | ORG `भारत इलेक्ट्रोनिक्स लिमिटेड`, `हिन्दुस्तान एयरोनोटिक्स लिमिटेड`, `एल एंड टी डिफेंस` (company names) | nothing |
| type confusion | LOC `यूपी`, `पालघर`, `वेस्टइंडीज`, `आसियान` | ORG |
| type confusion | ORG `विधानसभा`, `संघ`; ORG `इनेलो`, `डीपीएस`; PER `वाजपेयी` | LOC; PER; ORG |

1. **Boundary errors are mostly over-long spans.** In 276 of the 317 boundary errors the prediction is longer than gold (37 shorter, 4 equal length). The model attaches a role (`गृह मंत्री`, `स्पिनर`, `डायरेक्टर`), a title (`श्री`), or a following word (`धरना`) to a name, or adds `पार्टी` to a party. Gold names are bare. 61 of the 317 add one of ten common title or institution words (श्री, मंत्री, प्रधानमंत्री, राष्ट्रपति, नेता, अध्यक्ष, डॉ, सुश्री, पार्टी, सरकार). It also merges a name with its acronym in brackets, where gold keeps two entities.
2. **False alarms are generic words,** not names: countries and states as common nouns (`देश`, `प्रदेश`), job titles read as persons (`राष्ट्रपति`, `राज्यपाल`), and institution nouns (`पुलिस`, `मंत्रालय`, `विधानसभा`, `बैंक`). The same word is sometimes gold ORG (`विधानसभा`) and sometimes not, so part of the spurious count is gold inconsistency.
3. **Misses include plain place names** (`भारत`, `ईरान`, `अमेरिका`), abbreviations (`हप्र`) and several company names. That is surprising for a 7B model and this analysis does not explain it.
4. **Type confusion is mostly LOC read as ORG** (21), then ORG read as LOC (13). MuRIL's largest confusion is the same pair (LOC as ORG, 15), so part of it comes from gold conventions (for example `विधानसभा` as ORG, `यूपी` as LOC), not only from the LLM.

## 3. MuRIL (for comparison)

Spurious ORG (94) is MuRIL's largest cell, then boundary ORG (64). Examples: `आईपीएल`, `एनआरसी`, `इंस्टाग्राम`, `मेक इन इंडिया`, `स्वच्छ भारत मिशन` tagged ORG where gold has nothing; gold `वन मंत्रालय` vs `मंत्रालय`, `केएसके महानदी पावर` vs `... लिमिटेड`; missed `ट्विटर` (3), `लोकसभा`, `संसद`. Some gold labels look odd (ORG for `दिल्ली`, `अन्ना`, `वाजपेयी`), so part of the ceiling is annotation, not modelling. These gold examples were read after the prompt was frozen and did not change it.

## 4. LLM format drift

- **Malformed: 11 of 864 zero-shot (1.27%), 5 of 864 few-shot (0.58%).** Six zero-shot and all five few-shot failures were answers cut off at the 320-token limit. The other five zero-shot failures were short answers with broken JSON, so a larger token limit would not fix them. Each was scored as all `O` and counted.
- **No other drift:** 0 answers wrapped in prose or code fences, 0 invalid items, 1 unknown label per mode.
- **The model does not always copy the sentence.** 32 zero-shot and 31 few-shot spans matched nothing in the sentence and were lost: altered names (`नई दिल्ली` where the sentence has `दिल्ली`, 3 and 2 times; `सलमान कांहन` for `सलमान खान`; `दिनाक भास्कर` and `डेनिक भास्कर` for `दैनिक भास्कर`; `द न्यॉर्क टाइम्स`), a few words with stray characters from other scripts, and in zero-shot two Latin-script fragments (`ERS TIMES`). Few-shot wrote no Latin-script spans on gold (on the validation sample it wrote three place names in Latin).

## 5. All Hindi stages

Source: [phase5-cross-stage-errors.ipynb](../../notebooks/phase5-cross-stage-errors.ipynb), commit `a53f15e`. All 19 system-language scores reproduced their logged F1 within 0.001, and the Phase 5 error counts reproduced exactly. Tables are in [results/error_analysis/](../../results/error_analysis/).

| System | Boundary | Missed | Spurious | Type confusion | Total |
| ------ | -------- | ------ | -------- | -------------- | ----- |
| MuRIL (seed 42) | 167 | 77 | 205 | 31 | **480** |
| BiLSTM-CRF | 173 | 197 | 148 | 33 | 551 |
| IndicBERT | 204 | 160 | 204 | 32 | 600 |
| LLM few-shot | 317 | 227 | 406 | 53 | 1,003 |
| LLM zero-shot | 293 | 299 | 478 | 52 | 1,122 |

- **MuRIL's gain over BiLSTM-CRF is recall.** It misses 77 entities against 197, but makes more spurious predictions (205 against 148).
- **Most entities are easy, and ORG is over-represented among the hard ones.** Of 1,929 gold entities, 1,023 are right in BiLSTM-CRF, MuRIL, IndicBERT, and LLM few-shot, and 121 are wrong in all four. Those 121 are 54 ORG (10.3% of ORG), 36 LOC (5.9%), and 31 PER (3.9%).

## 6. Marathi and Bengali

Mean errors across seeds 42, 43, and 44. Change = in-language minus zero-shot, paired by seed, so a negative value means the in-language model makes fewer errors.

| Target | Kind | Hindi model | In-language | Change | Per seed |
| ------ | ---- | ----------- | ----------- | ------ | -------- |
| Marathi | **Total** | 393.0 | 435.3 | **+42.3** | +37 / +47 / +43 |
| Marathi | Missed | 82.7 | 128.3 | **+45.7** | +47 / +43 / +47 |
| Marathi | Spurious | 125.7 | 113.3 | **−12.3** | −16 / −9 / −12 |
| Marathi | Boundary | 159.7 | 166.7 | +7.0 | +6 / +13 / +2 |
| Bengali | **Total** | 284.3 | 270.3 | **−14.0** | −22 / −10 / −10 |
| Bengali | Missed | 90.3 | 78.7 | **−11.7** | −11 / −12 / −12 |
| Bengali | Spurious | 83.3 | 85.0 | +1.7 | −3 / +4 / +4 |
| Bengali | Boundary | 93.0 | 88.7 | −4.3 | −7 / +2 / −8 |

- **Marathi:** the in-language models are more conservative. They miss about 46 more entities and make about 12 fewer spurious predictions. The extra misses are mostly PER (+22.7) and LOC (+15.0).
- **Bengali:** in-language training mainly reduces misses, especially PER (−11.3). Its boundary and spurious changes are not consistent across seeds.
- **Consistent wins** are entities all three models on one side get right and all three on the other get wrong. On Marathi, Hindi models win 50 and in-language models win 12, out of 1,676 entities. On Bengali, they win 7 and 18, out of 997.
- **Examples:** Marathi entities won by Hindi models include `नाशिक` and `राष्ट्रवादी काँग्रेस`; those won by in-language models include `नागपूरच्या` and `भाजपच्या`. Bengali entities won by in-language models include `চীনের` and `ন্যাশনাল জিওগ্রাফিক`.

## 7. Script and tokenizer effects

| Language | Mean MuRIL subwords per word | Words with multiple subwords | Words with `[UNK]` |
| -------- | ---------------------------- | ---------------------------- | ------------------ |
| Hindi | 1.124 | 8.19% | 0.015% |
| Marathi | 1.315 | 20.35% | 0.008% |
| Bengali | 1.305 | 18.09% | 0.046% |

- **Tokenizer coverage does not appear to explain Bengali errors.** `[UNK]` is rare, affecting only 2 of 997 Bengali entities, and Bengali needs about as many subwords per word as Marathi.
- **Native-script entity recall** is 0.8412 (Hindi model) and 0.8098 (in-language) on Marathi, and 0.7992 and 0.8165 on Bengali.
- **Long entities are hardest.** Entities with 5 or more subwords have the lowest recall in every language and model group.
- **Limits:** Latin-script entities are too few to interpret (13 Hindi, 8 Marathi, 12 Bengali). Each language has a different test set, so these recalls show patterns rather than a direct comparison. This analysis does not show why Marathi in-language recall is lower. Silver label quality remains untested.
