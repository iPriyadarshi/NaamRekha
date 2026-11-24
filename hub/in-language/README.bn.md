---
language:
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
widget:
  - text: "রবীন্দ্রনাথ ঠাকুর কলকাতায় জন্মগ্রহণ করেন এবং বিশ্বভারতী বিশ্ববিদ্যালয় প্রতিষ্ঠা করেন।"
---

# NaamRekha MuRIL (Bengali NER, in-language baseline)

MuRIL fine-tuned on 300,000 Bengali sentences from [Naamapadam](https://huggingface.co/datasets/ai4bharat/naamapadam) to tag PER, ORG and LOC (IOB2). Seed 42.

It is the in-language baseline for the NaamRekha transfer study. On the Bengali gold test set it scores entity F1 **0.8136** (PER 0.8621, ORG 0.7233, LOC 0.8018; 3-seed mean 0.8132). That is ahead of the Hindi-trained [`priyadarshi-su/naamrekha-muril-hi`](https://huggingface.co/priyadarshi-su/naamrekha-muril-hi) used zero-shot (3-seed mean 0.8037), so prefer this model for Bengali.

Same setup as the Hindi model: lr 2e-5, batch size 32, 3 epochs, `max_length=128`. Use with `pipeline("token-classification", model=..., aggregation_strategy="first")`.

Project: [code](https://github.com/iPriyadarshi/NaamRekha) · [website](https://ipriyadarshi.github.io/NaamRekha/) · [demo](https://huggingface.co/spaces/priyadarshi-su/naamrekha-ner)
