---
pretty_name: TaDSE Top-2 Dialogue Pairs
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
size_categories:
- 100K<n<1M
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
# TaDSE Top-2 Dialogue Pairs

Private review release of the **exact 785,870 training pairs** used by both
parents of [the merged TaDSE BERT release](https://huggingface.co/minsik-oh/tadse-bert-base-uncased).
It contains original and synthetically augmented utterance-template pairs.
"Top-2" refers to SNIPS; other sources retain their recovered top-k settings.

This is the all-intent SNIPS top-2 variant, **not the paper's legacy top-5 corpus**.
No new generation, NER, balancing, truncation or deduplication is performed during
packaging. Pair multiplicity, text, embedded newlines and training order are
preserved. Source metadata is added after the original three training columns.

## Load

```python
from datasets import load_dataset

pairs = load_dataset(
    "minsik-oh/tadse-top2-dialogue-pairs",
    "combined",
    split="train",
    token=True,
)
print(len(pairs))  # 785870
print(pairs[0])
```

Authenticate with an account authorized for this private repository. Use config
`snips`, `atis`, `massive`, `hwu` or `clinc` for one source. There is only a
`train` split; evaluation datasets are not packaged here. Pin the Hub revision
from the release manifest for reproducibility.

## Contents and paper comparison

| Source | Top-k | Released pairs | Paper Table 1 utterances |
| --- | ---: | ---: | ---: |
| SNIPS, all seven intents | 2 | 115,378 | 163K, top-5 |
| ATIS | 2 | 238,847 | 239K |
| MASSIVE, en-US | 3 | 78,289 | 78K |
| HWU | 3 | 133,176 | 133K |
| CLINC | 5 | 220,180 | 220K |
| **Total** | — | **785,870** | **834K** |

The four non-SNIPS counts round to the displayed paper counts. SNIPS deliberately
uses top-2 and retains all intents, unlike the recovered legacy output overwrite
that kept only its final intent. Released counts are logical training records,
not physical CSV lines. Numerical paper parity is not claimed for this variant.

SNIPS pairs by intent: AddToPlaylist 15,975; BookRestaurant 28,663; GetWeather
13,644; PlayMusic 13,276; RateBook 22,412; SearchCreativeWork 8,792;
SearchScreeningEvent 12,616. These follow the original generator's intent order.

## Columns

| Column | Meaning |
| --- | --- |
| `utterance` | Exact original or augmented utterance used for training |
| `pattern` | Exact corresponding template, with generic `{SLOT}` markers |
| `slot_count` | Number of `{SLOT}` markers in the pattern |
| `source` | `snips`, `atis`, `massive`, `hwu` or `clinc` |
| `source_row` | Zero-based position within that source's training CSV |

Intent labels and original-versus-synthetic flags are not supplied: those were
not retained as columns in the training artifact. `source_row` preserves record
identity even when text pairs repeat. `combined` concatenates sources in this
order: SNIPS, ATIS, MASSIVE, HWU, CLINC.

## Provenance and verification

The original augmentation script was recovered byte-for-byte from private Git
commit `48d5f94`; its SHA256 is
`d7f3daa57b58e79c391d86caa9e5cb1aebc8b98f44188dabef2307a59a1636a6`.
The top-2 adapter preserves all SNIPS intent writes in the trainable CSV without
changing the generator's augmentation functions. CLINC reuses cached historical
spaCy annotations; the other datasets use supplied annotations. No NER is run
as part of packaging or evaluation.

The input combined CSV SHA256 is
`9b8404042073814afbf0a63670e8ef9919d722cf8bee1b7aea208ecdad696dd4`.
`provenance.json` preserves the generation manifest, input hashes and per-intent
counts. `verification.json` records the CSV/content/Parquet checks and file hashes.
Every packaged row is checked against the source CSV and the original combined
training order. No checkpoint, credential, environment or unrelated file is
included.

Historical decoding, duplicate pairs and source overlap are retained to match
the trained model. This is not a newly decontaminated dataset. The exact original
paper training file and launch configuration remain unavailable. Source datasets
retain their respective licenses and attribution requirements; this private
package does not assert a new blanket license.

Paper: [Template-assisted Contrastive Learning of Task-oriented Dialogue Sentence Embeddings](https://arxiv.org/abs/2305.14299)

Source code: https://github.com/minsik-ai/Template-Contrastive-Embedding
