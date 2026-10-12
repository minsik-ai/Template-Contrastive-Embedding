# TaDSE: Template-aware Dialogue Sentence Embeddings

Paper: [Template-assisted Contrastive Learning of Task-oriented Dialogue Sentence Embeddings](https://arxiv.org/abs/2305.14299)

Accepted to **ACL 2026 (Oral)**.

TaDSE learns sentence embeddings for spoken/dialogue language understanding by
jointly contrasting utterances and their underlying **patterns** (slot-templated
forms of the utterance, e.g. `play [artist:Adele]` → `play [artist]`). It extends
SimCSE with an additional pattern-contrastive objective and a pairwise
utterance↔pattern alignment loss.

## Released model and dataset

- [Template Dialogue Embedding](https://huggingface.co/minsik-oh/template-dialogue-embedding): a BERT encoder for English dialogue utterances.
- [Template Dialogue Pairs](https://huggingface.co/datasets/minsik-oh/template-dialogue-pairs): the utterance–template training data used for the released model.

The model works with standard Transformers. Use the exported **pooler output**
and normalize it for cosine similarity; the template projection is training-only.

```python
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

model_id = "minsik-oh/template-dialogue-embedding"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id).eval()
tokens = tokenizer(["Book a flight to Boston", "Find flights to Boston"],
                   padding=True, truncation=True, max_length=128,
                   return_tensors="pt")
with torch.inference_mode():
    embeddings = F.normalize(model(**tokens).pooler_output, p=2, dim=-1)
```

For training, head export, merging and recovered evaluation, follow the
[released-model reproduction guide](docs/released-model.md). The release is a
newly trained checkpoint; it is not the original paper checkpoint.

## Repository layout

```
data/massive/          Bundled source datasets and preprocessing
legacy/
  augment_data.py     Recovered augmentation source
  accuracy_infer.py   Recovered evaluation source
  simcse_tool.py      Recovered embedding/search implementation
  training/           Original entry point, custom CLTrainer, model and collator
scripts/              Portable release preparation, training, export and evaluation
model/
  tadse/              Earlier compatibility models and experimental adaptations
  supervised/         Downstream JointBERT-style intent classification
eval/                 Alignment/uniformity and representation utilities
docs/released-model.md   Published model reproduction guide
releases/             Model/dataset cards and verified manifests
requirements.txt      Modern preparation/runtime dependencies
```

## Method

Each training instance yields four views per example after tokenization and are
stacked along `num_sent=4`: `[utt1, utt2, ptn1, ptn2]` (two dropout views of the
utterance and two of its pattern). `cl_forward` in `model/tadse/models.py`
computes three losses:

- `simcse_utt_loss` — SimCSE InfoNCE between `utt1` and `utt2`
- `simcse_ptn_loss` — SimCSE InfoNCE between `ptn1` and `ptn2`
- `pairwise_loss`   — cross-modal InfoNCE between `utt1` and `ptn1`
  (configurable via `pairwise_neg_type`: `pattern`, `utterance`, or `balanced`)

Total loss:
```
L = α · simcse_utt_loss + β · simcse_ptn_loss + γ · pairwise_loss  (+ optional MLM)
```
with ratios `simcse_utt_loss_ratio`, `simcse_ptn_loss_ratio`,
`pairwise_loss_ratio` on `ModelArguments`. An optional `PatternMLPLayer`
(`apply_ptn_transform`) projects pattern embeddings into the utterance space.

Backbones: BERT, RoBERTa, and Qwen3 (`BertForPatternCL`, `RobertaForPatternCL`,
`Qwen3ForPatternCL`). Pooling is controlled by `pooler_type`
(`cls`, `cls_before_pooler`, `pad_left_no_pooler`, `avg`, `avg_top2`,
`avg_first_last`).

## Setup and reproduction

The release uses the **recovered original training stack**, including the custom
`CLTrainer`, model, collator and entry point preserved under `legacy/training/`.
The [release guide](docs/released-model.md) gives portable commands to obtain the
published training data, train both parent models, export their trained heads,
and produce the fixed merged encoder. It separates the Python 3.11 preparation
runtime from the Python 3.8 / Transformers 4.2.1 training runtime.

Recovered components and their adapters:

- `legacy/augment_data.py` — preserved augmentation source.
- `scripts/prepare_table2_training.py` — replays that source with explicit
  dataset settings and provenance; CLINC reuses its cached spaCy annotations.
- `legacy/training/` — original custom training loop and TaDSE objectives.
- `scripts/run_simcse_head_ablation.py` — initializes the shared contrastive head
  from the pretrained SimCSE pooler before the original training entry point runs.
- `scripts/export_recovered_training.py` — exports the trained utterance head into
  a standalone BERT pooler and checks exact reload equivalence.
- `scripts/merge_exported_models.py` — applies the fixed equal-weight merge.
- `scripts/evaluate_recovered.py` — recovered utterance-only nearest-neighbour
  evaluation, including the original HWU selection and CLINC OOS examples.

The earlier reconstruction in `docs/reproduction.md`, the stock-Trainer runner
`scripts/reproduce_train.py`, and older experiment notes remain as historical
records. Use the release guide for the published model's recipe. The original
historical launch command and complete environment lock were not recovered;
re-running the released recipe does not establish exact reproduction of the
paper's checkpoints or results.

## Downstream intent classification

`model/supervised` is a JointBERT-style trainer for MASSIVE/ATIS/SNIPS.

```bash
cd model/supervised
python main.py \
  --task massive --model_type bert \
  --model_dir runs/intent-bert \
  --data_dir ../../data/massive/joint \
  --do_train --do_eval
```

## Evaluation utilities

- `eval/calc_align_uniform.py` — alignment/uniformity of the learned space.
- `eval/repr_tsne.py` — t-SNE plots of utterance vs pattern representations.

## Datasets

Bundled under `data/massive/`: MASSIVE (`massive_en-US.jsonl`, joint splits),
ATIS (`atis_train.json`, `atis_test.json`), SNIPS (per-intent folders),
HWU64 (raw CSVs) and CLINC pattern extraction logs.

## Citation

```bibtex
@inproceedings{oh-etal-2026-template,
    title = "Template-assisted Contrastive Learning of Task-oriented Dialogue Sentence Embeddings",
    author = "Oh, Minsik  and
      Li, Jiwei  and
      Wang, Guoyin",
    editor = "Liakata, Maria  and
      Moreira, Viviane P.  and
      Zhang, Jiajun  and
      Jurgens, David",
    booktitle = "Proceedings of the 64th Annual Meeting of the {A}ssociation for {C}omputational {L}inguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2026",
    address = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.acl-long.1015/",
    doi = "10.18653/v1/2026.acl-long.1015",
    pages = "22181--22198",
    ISBN = "979-8-89176-390-6",
    abstract = "Learning high quality sentence embeddings from dialogues has drawn increasing attentions as it is essential to solve a variety of dialogue-oriented tasks with low annotation cost. Annotating and gathering utterance relationships in conversations are difficult, while token-level annotations, , entities, slots and templates, are much easier to obtain. Other sentence embedding methods are usually sentence-level self-supervised frameworks and cannot utilize token-level extra knowledge. We introduce Template-aware Dialogue Sentence Embedding (TaDSE), a novel augmentation method that utilizes template information to learn utterance embeddings via self-supervised contrastive learning framework. We further enhance the effect with a synthetically augmented dataset that diversifies utterance-template association, in which slot-filling is a preliminary step. We evaluate TaDSE performance on five downstream benchmark dialogue datasets. The experiment results show that TaDSE achieves significant improvements over previous SOTA methods for dialogue. We further introduce a novel analytic instrument of semantic compression test, for which we discover a correlation with uniformity and alignment. Our code is available at \url{https://github.com/minsik-ai/Template-Contrastive-Embedding}"
}
```
