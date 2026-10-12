---
pretty_name: Template Dialogue Pairs
language:
- en
task_categories:
- sentence-similarity
tags:
- tadse
- dialogue
- contrastive-learning
- synthetic
- template-augmentation
configs:
- config_name: combined
  default: true
  data_files:
  - split: train
    path:
    - data/snips-train.parquet
    - data/atis-train.parquet
    - data/massive-train.parquet
    - data/hwu-train.parquet
    - data/clinc-train.parquet
- config_name: snips
  data_files:
  - split: train
    path: data/snips-train.parquet
- config_name: atis
  data_files:
  - split: train
    path: data/atis-train.parquet
- config_name: massive
  data_files:
  - split: train
    path: data/massive-train.parquet
- config_name: hwu
  data_files:
  - split: train
    path: data/hwu-train.parquet
- config_name: clinc
  data_files:
  - split: train
    path: data/clinc-train.parquet
---
# Template Dialogue Pairs

English utterance-template pairs for dialogue representation learning,
combining original and synthetically augmented examples from five datasets.
This is the training corpus used by [Template Dialogue Embedding](https://huggingface.co/minsik-oh/template-dialogue-embedding).

## Load

```python
from datasets import load_dataset

pairs = load_dataset(
    "minsik-oh/template-dialogue-pairs",
    "combined",
    split="train",
    token=True,
)
print(pairs[0])
```

While the repository is private, authenticate with an authorized account. Use
config `snips`, `atis`, `massive`, `hwu` or `clinc` for one source. Only the
`train` split is included. Pin a Hub revision for reproducibility.

## Contents

The corpus combines SNIPS (all seven intents), ATIS, MASSIVE (en-US), HWU
and CLINC, including original and synthetically augmented examples.

## Columns

| Column | Meaning |
| --- | --- |
| `utterance` | Original or augmented utterance used for training |
| `pattern` | Corresponding template, with generic `{SLOT}` markers |
| `slot_count` | Number of `{SLOT}` markers in the pattern |
| `source` | `snips`, `atis`, `massive`, `hwu` or `clinc` |
| `source_row` | Zero-based position within that source's training CSV |

Intent labels and original-versus-synthetic flags are not included. `source_row`
preserves record identity when text pairs repeat. The `combined` configuration
concatenates SNIPS, ATIS, MASSIVE, HWU and CLINC in that order.

## Preparation and verification

Templates use supplied slot annotations for SNIPS, ATIS, MASSIVE and HWU, and
cached spaCy NER annotations for CLINC. The release retains all seven SNIPS
intents. Packaging preserves the training text, encoding, embedded newlines,
duplicates and order without further generation or normalization. Existing
source overlap is retained; this is not a newly decontaminated dataset.

Every Parquet row was checked against the training CSVs. The complete dataset
was also loaded from Hugging Face and checked for matching content and order.
`provenance.json` records generation settings, input hashes and per-intent counts;
`verification.json` records packaging checks. These files preserve detailed
provenance beyond this release overview.

Combined training CSV SHA256:
`9b8404042073814afbf0a63670e8ef9919d722cf8bee1b7aea208ecdad696dd4`.

## Sources and attribution

The source datasets retain their respective licenses and attribution requirements;
this package does not assert a new blanket license. SNIPS attribution: Coucke,
Alice, et al. (2018), *Snips Voice Platform: an embedded Spoken Language
Understanding system for private-by-design voice interfaces*,
[arXiv:1805.10190](https://arxiv.org/abs/1805.10190).

Paper: [Template-assisted Contrastive Learning of Task-oriented Dialogue Sentence Embeddings](https://arxiv.org/abs/2305.14299)

Source code: https://github.com/minsik-ai/Template-Contrastive-Embedding
