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
# Template Dialogue Embedding

TaDSE produces **768-dimensional embeddings** for dialogue utterances. This
release combines two TaDSE checkpoints with a fixed 50/50 weight average and
runs as a single standard BERT encoder. Use **`pooler_output`** for embeddings.

The two checkpoints were trained with and without an additional template
projection. That projection is used only during training. The released model
contains the averaged encoder and trained utterance head, with no custom
inference code or additional forward pass required.

## Usage

```python
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

model_id = "minsik-oh/template-dialogue-embedding"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id).eval()
tokens = tokenizer(["Book a table for two", "Reserve a restaurant"],
                   padding=True, truncation=True, max_length=128,
                   return_tensors="pt")
with torch.inference_mode():
    embeddings = F.normalize(model(**tokens).pooler_output, p=2, dim=-1)
print(embeddings @ embeddings.T)
```

Pin a Hub revision for reproducibility. Mean pooling is not the evaluated
representation.

## Evaluation

Intent classification uses utterance-only cosine nearest-neighbour retrieval
against each dataset's training references. The same released model is used
for every dataset.

Evaluation uses the recovered utterance-only protocol. HWU uses raw-file split
matching and CLINC includes out-of-scope queries; existing source overlaps are
retained. Detailed single-seed results and settings are available in
`evaluation.json`. The merge ratio was fixed before evaluation, with no
per-dataset model selection.

## Training

Training data: [Template Dialogue Pairs](https://huggingface.co/datasets/minsik-oh/template-dialogue-pairs),
utterance-template pairs from SNIPS, ATIS, MASSIVE, HWU and CLINC.
Detailed generation settings are recorded in the dataset provenance.

Both checkpoints initialize from `princeton-nlp/unsup-simcse-bert-base-uncased`,
revision `218baf6bc48c463a53f5b4e1c568299537fab124`, including its trained shared
contrastive head. Training uses two epochs, batch 16, learning rate 1e-8,
maximum length 32, seed 42 and FP32. Utterance/template/pairwise loss weights
are 1/1/0.5, with utterance negatives. The implementation uses the original
TaDSE custom trainer with Transformers 4.2.1 and PyTorch 2.2.2+cu121.

## Provenance

`training-recipe.json` records the release recipe. `merge-audit.json` identifies
the checkpoint sources and hashes. `serialization-audit.json` verifies that
safetensors conversion preserves model tensors and probe outputs exactly.
Detailed reconstruction records remain in the source repository.

Paper: [Template-assisted Contrastive Learning of Task-oriented Dialogue Sentence Embeddings](https://arxiv.org/abs/2305.14299)

Source: https://github.com/minsik-ai/Template-Contrastive-Embedding
