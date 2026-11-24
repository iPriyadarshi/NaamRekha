---
language:
  - hi
  - mr
  - bn
license: apache-2.0
library_name: transformers
pipeline_tag: token-classification
base_model: google/muril-base-cased
datasets:
  - ai4bharat/naamapadam
tags:
  - ner
  - named-entity-recognition
  - indic
  - cross-lingual
widget:
  - text: "प्रधानमंत्री नरेंद्र मोदी ने नई दिल्ली में भारतीय रिज़र्व बैंक के अधिकारियों से मुलाकात की।"
model-index:
  - name: naamrekha-muril-hi
    results:
      - task:
          type: token-classification
          name: Named Entity Recognition
        dataset:
          type: ai4bharat/naamapadam
          name: Naamapadam (Hindi, gold test)
          config: hi
          split: test
        metrics:
          - type: f1
            value: 0.8274
            name: Entity F1 (seqeval)
      - task:
          type: token-classification
          name: Named Entity Recognition (zero-shot)
        dataset:
          type: ai4bharat/naamapadam
          name: Naamapadam (Marathi, gold test)
          config: mr
          split: test
        metrics:
          - type: f1
            value: 0.8298
            name: Entity F1 (seqeval)
      - task:
          type: token-classification
          name: Named Entity Recognition (zero-shot)
        dataset:
          type: ai4bharat/naamapadam
          name: Naamapadam (Bengali, gold test)
          config: bn
          split: test
        metrics:
          - type: f1
            value: 0.7982
            name: Entity F1 (seqeval)
---

# NaamRekha MuRIL (Hindi NER)

A Hindi named entity recognition model that tags **person (PER)**, **organisation (ORG)** and **location (LOC)** names. It is [MuRIL](https://huggingface.co/google/muril-base-cased) fine-tuned on 300,000 Hindi sentences from [Naamapadam](https://huggingface.co/datasets/ai4bharat/naamapadam).

Although it was trained only on Hindi, it also works **zero-shot on Marathi and Bengali**. On Marathi it beats a MuRIL model trained on Marathi itself.

Part of the NaamRekha project: [code](https://github.com/iPriyadarshi/NaamRekha) · [website](https://ipriyadarshi.github.io/NaamRekha/) · [report](https://ipriyadarshi.github.io/NaamRekha/docs/phase6/report/) · [demo](https://huggingface.co/spaces/priyadarshi-su/naamrekha-ner)

## Usage

```python
from transformers import pipeline

ner = pipeline(
    "token-classification",
    model="priyadarshi-su/naamrekha-muril-hi",
    aggregation_strategy="first",  # the model was trained with labels on each word's first subword
)
ner("प्रधानमंत्री नरेंद्र मोदी ने नई दिल्ली में भारतीय रिज़र्व बैंक के अधिकारियों से मुलाकात की।")
```

Labels are IOB2: `O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC`.

## Results

Entity-level micro F1 with `seqeval` on Naamapadam's human-annotated gold test sets. Training used only Hindi.

| Test language | PER | ORG | LOC | Overall (this model) | Overall, 3-seed mean | In-language MuRIL, 3-seed mean |
| ------------- | --- | --- | --- | -------------------- | -------------------- | ------------------------------ |
| Hindi | 0.8879 | 0.7444 | 0.8224 | **0.8274** | 0.8230 | (same) |
| Marathi (zero-shot) | 0.8974 | 0.6866 | 0.8457 | **0.8298** | 0.8332 | 0.8163 |
| Bengali (zero-shot) | 0.8506 | 0.6963 | 0.7906 | **0.7982** | 0.8037 | 0.8132 |

This checkpoint is seed 42, the seed fixed in advance as the project's handoff model. On Hindi it beats a BiLSTM-CRF baseline (0.8038), untuned IndicBERT (0.7839), and few-shot prompting of Qwen2.5-7B (0.6643). Every number comes from the project's [results.csv](https://github.com/iPriyadarshi/NaamRekha/blob/main/results/results.csv).

## Training

- **Base model:** `google/muril-base-cased`, with `AutoModelForTokenClassification`.
- **Data:** a seeded 300,000-sentence subset of Naamapadam's Hindi **silver** training set (labels projected from English NER). Model and size choices were made on silver validation; the gold test set was used only for reporting.
- **Hyperparameters:** lr 2e-5, batch size 32, 3 epochs, linear warmup over the first 10% of steps, `max_length=128`, fp16, one T4 GPU, seed 42.
- **Label alignment:** each word's label goes on its first subword; other subwords get `-100`.
- **Software:** torch 2.11.0, transformers 5.22.1.

The weights are identical to [`priyadarshi-su/naamrekha-muril_hi_n300000_e3_lr2e-05_s42_bfd254ed`](https://huggingface.co/priyadarshi-su/naamrekha-muril_hi_n300000_e3_lr2e-05_s42_bfd254ed/tree/de5204680ed4a8ab1ac842847df6866a6e2c9251) at revision `de52046`, the checkpoint cited in the project report.

## Limitations

- **Silver training labels.** Training labels were projected from English, so they are noisy. Scores are on gold test sets.
- **ORG is the weakest type** in every language, and the main source of errors.
- **Three entity types only.** No dates, numbers, or other categories.
- **Sentence-length inputs.** Trained on sentences up to 128 subword tokens; split long text into sentences first.
- **Domain.** Naamapadam sentences come from a parallel corpus, mostly news-like text. Expect lower accuracy on social media, code-mixed, or romanised text.
- **Zero-shot Bengali** trails a Bengali-trained model by about 0.01 F1. Use an in-language model where one exists.

## Citation

If you use this model, please cite the dataset and base model:

- Mhaske et al., *Naamapadam: A Large-Scale Named Entity Annotated Data for Indic Languages*, ACL 2023.
- Khanuja et al., *MuRIL: Multilingual Representations for Indian Languages*, 2021.
