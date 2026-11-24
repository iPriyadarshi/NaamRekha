---
language:
  - mr
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
  - text: "सचिन तेंडुलकर यांनी मुंबई येथे भारतीय क्रिकेट नियामक मंडळाच्या बैठकीत भाग घेतला."
---

# NaamRekha MuRIL (Marathi NER, in-language baseline)

MuRIL fine-tuned on 300,000 Marathi sentences from [Naamapadam](https://huggingface.co/datasets/ai4bharat/naamapadam) to tag PER, ORG and LOC (IOB2). Seed 42.

It is the in-language baseline for the NaamRekha transfer study. On the Marathi gold test set it scores entity F1 **0.8156** (PER 0.8902, ORG 0.6448, LOC 0.8437; 3-seed mean 0.8163). The Hindi-trained [`priyadarshi-su/naamrekha-muril-hi`](https://huggingface.co/priyadarshi-su/naamrekha-muril-hi) does better on Marathi zero-shot (3-seed mean 0.8332), so prefer that model for Marathi.

Same setup as the Hindi model: lr 2e-5, batch size 32, 3 epochs, `max_length=128`. Use with `pipeline("token-classification", model=..., aggregation_strategy="first")`.

Project: [code](https://github.com/iPriyadarshi/NaamRekha) · [website](https://ipriyadarshi.github.io/NaamRekha/) · [demo](https://huggingface.co/spaces/priyadarshi-su/naamrekha-ner)
