# TaDSE: Template-aware Dialogue Sentence Embeddings

Paper: [Template-assisted Contrastive Learning of Task-oriented Dialogue Sentence Embeddings](https://arxiv.org/abs/2305.14299)

Accepted to **ACL 2026 (Oral)**.

TaDSE learns sentence embeddings for spoken/dialogue language understanding by
jointly contrasting utterances and their underlying **patterns** (slot-templated
forms of the utterance, e.g. `play [artist:Adele]` → `play [artist]`). It extends
SimCSE with an additional pattern-contrastive objective and a pairwise
utterance↔pattern alignment loss.

## Repository layout

```
data/massive/          Datasets + preprocessing (MASSIVE, ATIS, SNIPS, CLINC, HWU64)
  proc_data.py         Data processing utilities
  augment_data.py      Pattern/augmentation generation
  extract_ptn_clinc.py Pattern extraction (CLINC, via spaCy NER)
  simcse/              Vendored SimCSE tool for sentence embedding utilities
  joint/               MASSIVE joint intent+slot splits (train/dev/test)
  snips/               SNIPS intent data
model/
  tadse/
    models.py          BertForPatternCL / RobertaForPatternCL / Qwen3ForPatternCL
                       + cl_init / cl_forward (utterance+pattern contrastive loss)
    trainers.py        HuggingFace Trainer subclass (CLTrainer)
  pairwise_pattern_train.py   Pre-training entry point (HF-style args)
  supervised/          Downstream JointBERT-style intent classification
    main.py, trainer.py, data_loader.py, utils.py
eval/
  calc_align_uniform.py   Alignment/uniformity metrics (Wang & Isola)
  repr_tsne.py            t-SNE visualization of learned representations
requirements.txt
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

## Setup

```bash
pip install -r requirements.txt
```

## Pre-training TaDSE

```bash
python model/pairwise_pattern_train.py \
  --model_name_or_path bert-base-uncased \
  --train_file data/massive/... \
  --output_dir runs/tadse-bert \
  --do_train
```
See `ModelArguments` / `DataTrainingArguments` in
`model/pairwise_pattern_train.py` for all flags (loss ratios, temperatures,
pooler type, pattern transform, etc.).

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
