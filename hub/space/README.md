---
title: NaamRekha NER
emoji: 🏷️
colorFrom: indigo
colorTo: blue
sdk: static
app_file: index.html
pinned: false
license: apache-2.0
short_description: Find people, places and orgs in Hindi, Marathi, Bengali
models:
  - priyadarshi-su/naamrekha-muril-hi
  - priyadarshi-su/naamrekha-muril_mr_n300000_e3_lr2e-05_s42_36701790
  - priyadarshi-su/naamrekha-muril_bn_n300000_e3_lr2e-05_s42_46e8e677
datasets:
  - ai4bharat/naamapadam
---

# NaamRekha NER demo

Tag person, organisation and location names in Hindi, Marathi and Bengali with MuRIL models fine-tuned on Naamapadam.

The default model is trained only on Hindi and runs **zero-shot** on Marathi and Bengali. Switch to an in-language model to compare.

The models run in your browser with [transformers.js](https://huggingface.co/docs/transformers.js), using int8-quantized ONNX exports (about 240 MB each, downloaded once and cached). The scores on the model cards are for the full-precision PyTorch weights.

Project: [code](https://github.com/iPriyadarshi/NaamRekha) · [website](https://ipriyadarshi.github.io/NaamRekha/) · [model](https://huggingface.co/priyadarshi-su/naamrekha-muril-hi)
