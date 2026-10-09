---
language: en
library_name: transformers
pipeline_tag: feature-extraction
base_model: princeton-nlp/unsup-simcse-bert-base-uncased
tags:
- tadse
- sentence-embeddings
- contrastive-learning
- bert
---
# TaDSE BERT base — reproduced, trained-head export

Private review candidate for [Template-assisted Contrastive Learning of Task-oriented Dialogue Sentence Embeddings](https://arxiv.org/abs/2305.14299).
This is a new reproduction checkpoint, not the archived original paper checkpoint.
The non-pattern-MLP TaDSE model was trained for two epochs and its trained utterance
MLP was exported into the standard BERT pooler. Use `pooler_output`.

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
Hub commit revision recorded in the source repository release manifest for reproducibility.
No custom model code is required.

## Evaluation

Full-reference cosine nearest-neighbor intent accuracy, fixed exported head:

| Dataset | Accuracy (%) |
| --- | ---: |
| SNIPS | 97.57 |
| ATIS | 89.59 |
| MASSIVE | 77.64 |
| HWU64 | 81.97 |
| CLINC | 69.45 |
| Macro mean over five datasets | 83.24 |

The paper TaDSE mean is 83.64. This candidate does not claim exact replication.
Benchmarks retain historical overlaps; these are not newly decontaminated splits.
HWU uses canonical KFold1. CLINC includes out-of-scope queries.

## Training and provenance

Initialized from `princeton-nlp/unsup-simcse-bert-base-uncased` revision
`218baf6bc48c463a53f5b4e1c568299537fab124`. Two epochs, batch 16,
learning rate 1e-8, maximum training length 32, seed 42, FP32, utterance/pattern/
pairwise loss weights 1/1/0.5. No pattern transformation MLP.
Training contains 829,591 utterance-template pairs across SNIPS, ATIS, MASSIVE,
HWU and CLINC. It preserves the recovered SNIPS generator's output overwrite:
only SearchScreeningEvent survives. This model does **not** use the proposed
all-intent SNIPS top-2 corpus. Modern training dependencies and unrecovered
historical launch settings are additional reproduction limitations.

Export maps `mlp.dense` to `pooler.dense`; all other encoder tensors are identical.
CPU reload outputs match the source trained head exactly (maximum difference 0.0).
`export-audit.json` contains source and export checksums. Detailed evaluation and
training arguments are provided alongside the model. Local paths in provenance
files identify the reproducibility environment; they are not required for inference.

Source: https://github.com/minsik-ai/Template-Contrastive-Embedding
