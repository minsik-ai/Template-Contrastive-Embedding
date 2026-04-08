# TaDSE: Template-aware Dialogue Sentence Embeddings

Paper: [Template-assisted Contrastive Learning of Task-oriented Dialogue Sentence Embeddings](https://arxiv.org/abs/2305.14299) — accepted to **ACL 2026**.

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
@misc{oh2025templateassistedcontrastivelearningtaskoriented,
      title={Template-assisted Contrastive Learning of Task-oriented Dialogue Sentence Embeddings},
      author={Minsik Oh and Jiwei Li and Guoyin Wang},
      year={2025},
      eprint={2305.14299},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2305.14299},
}
```
