---
language: en
library_name: transformers
pipeline_tag: feature-extraction
base_model: princeton-nlp/unsup-simcse-bert-base-uncased
tags:
- tadse
- sentence-embeddings
- contrastive-learning
- model-merge
- bert
---
# TaDSE BERT base — top-2 merged model

Private release of a **50/50 weight average** of the exported top-2 TaDSE models,
with and without the separate training-time pattern MLP. This is one standard
BERT encoder with 768-dimensional embeddings. Use **`pooler_output`**.

This is a reproduced and merged model, not the archived original paper checkpoint.
Both parents start from the same pinned SimCSE backbone and initialize the shared
contrastive head from its pretrained pooler. Each was trained with the recovered
original TaDSE code and exported before averaging. The separate pattern MLP is
not part of either inference export. Encoder and pooler weights are averaged;
there is one forward pass per utterance, not a two-model ensemble.

## Usage

```python
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

model_id = "minsik-oh/tadse-bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id).eval()
tokens = tokenizer(["Book a table for two", "Reserve a restaurant"],
                   padding=True, truncation=True, max_length=128,
                   return_tensors="pt")
with torch.inference_mode():
    embeddings = F.normalize(model(**tokens).pooler_output, p=2, dim=-1)
print(embeddings @ embeddings.T)
```

Private access requires an authenticated account with repository access. Pin the
Hub revision recorded in the source repository release manifest for reproducibility.
No custom model code is required. Mean pooling is not the evaluated representation.

## Evaluation

Recovered utterance-only cosine nearest-neighbour intent evaluation, using all
reference examples and a fixed merged encoder for every dataset:

| Dataset | Test examples | SimCSE | Merged TaDSE |
| --- | ---: | ---: | ---: |
| SNIPS | 700 | 92.00 | 96.00 |
| ATIS | 893 | 85.67 | 89.59 |
| MASSIVE | 2,974 | 76.77 | 79.59 |
| HWU | 830 | 81.08 | 82.41 |
| CLINC | 5,500 | 71.00 | 72.25 |
| Mean over five datasets | — | 81.30 | **83.97** |

HWU uses recovered raw-file matching, not canonical KFold1. CLINC includes
out-of-scope queries. Historical data overlaps are retained, not newly filtered.
These are one-seed exploratory results; multi-seed robustness is not established.
The 50/50 ratio was fixed before merged evaluation. No per-dataset model selection
or ratio sweep was used. See `evaluation.json` for exact values.

## Training and provenance

Base: `princeton-nlp/unsup-simcse-bert-base-uncased`, revision
`218baf6bc48c463a53f5b4e1c568299537fab124`.
Both parents use the same 785,870 utterance-template pairs across SNIPS, ATIS,
MASSIVE, HWU and CLINC. SNIPS uses top-2 augmentation across all seven intents
(115,378 pairs); the other sources retain the recovered recipe. Two epochs,
batch 16, learning rate 1e-8, maximum training length 32, seed 42, FP32;
utterance/template/pairwise loss weights 1/1/0.5 with utterance negatives.

The original entry point, model and custom CLTrainer from private commit
`48d5f94` are preserved unchanged. Transformers 4.2.1 was restored; Torch was
historically unpinned, and these runs use 2.2.2+cu121. Exact historical launch
arguments and merged training-file bytes remain unavailable.

`merge-audit.json` identifies the two parent exports, their hashes and the fixed
merge. `serialization-audit.json` records conversion to safetensors with identical
model tensors and exact probe outputs. `training-recipe.json` records the recipe.
The previous private candidate remains accessible through Hub commit history.

Paper: [Template-assisted Contrastive Learning of Task-oriented Dialogue Sentence Embeddings](https://arxiv.org/abs/2305.14299)

Source: https://github.com/minsik-ai/Template-Contrastive-Embedding
