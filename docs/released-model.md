# Reproducing Template Dialogue Embedding

The public [model](https://huggingface.co/minsik-oh/template-dialogue-embedding)
and [training dataset](https://huggingface.co/datasets/minsik-oh/template-dialogue-pairs)
provide a runnable release recipe. The model is a fixed 50/50 parameter average
of two BERT encoders: TaDSE with and without the template MLP.
Both parents initialize their shared contrastive head from SimCSE. After training,
the utterance head is exported into each BERT pooler before merging. The released
model therefore needs one ordinary BERT forward pass and no template input.

Detailed settings, hashes and provenance are available in the Hub repositories
and the `releases/` manifests.

## Prepare the two environments

Run from the repository root on Linux with Git and `uv` available. Full training
uses a single CUDA GPU in FP32. The training runtime uses
Python 3.8.20, Torch 2.2.2 and Transformers 4.2.1. Keep it separate from the
Python 3.11 preparation environment:

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
bash scripts/setup_recovered_training.sh "$PWD"
```

The setup script creates an isolated training environment, installs pinned
dependencies and checks out SentEval at a fixed revision. The custom trainer
imports SentEval; this workflow uses the dialogue evaluation described below.

## Obtain the released training inputs

```bash
.venv/bin/python scripts/prepare_public_training.py \
  --output-dir artifacts/public-training
```

Use a fresh output directory. The helper downloads the published dataset and
pinned `princeton-nlp/unsup-simcse-bert-base-uncased` initialization. It reconstructs
`combined_train.csv` in the original training order, preserving text, duplicates
and CSV quoting, and verifies its content against the release.

The resulting `initial-model/` is the initialization checkpoint; it is not the
released TaDSE checkpoint. Training consumes `utterance`, `pattern`, and
`slot_count`; the dataset's `source` and `source_row` columns provide provenance.

## Train, export and merge

```bash
bash scripts/train_released_bert.sh \
  artifacts/public-training/combined_train.csv \
  artifacts/public-training/initial-model \
  artifacts/public-training/runs
```

Use a fresh run directory. The launcher runs both parents sequentially in
the training environment through the custom `CLTrainer`. Both use seed 42, two
epochs, batch size 16, learning rate `1e-8`, maximum training length 32, CLS pooling,
utterance negatives, and loss weights 1 / 1 / 0.5 for utterance contrastive,
template contrastive, and pairwise alignment losses. The second parent enables
`apply_ptn_transform`.

The initialization wrapper copies the pretrained SimCSE pooler into the shared
contrastive head and audits the changed tensors. The optional template MLP keeps
its own initialization. Export maps the trained utterance head into a standard
BERT pooler and checks exact reload outputs. The merge averages every floating
parameter, including the pooler, with fixed equal weights.

Outputs:

- `runs/nonmlp/export/` — exported parent without the template MLP.
- `runs/mlp/export/` — exported parent trained with the template MLP.
- `runs/merged/` — merged encoder and merge audit.

Use `runs/merged` with the inference example in the root README. Its
`pooler_output` is the embedding; normalize for cosine similarity. Retraining may
vary across hardware and software even with a fixed seed. The published weights
remain the reference artifact.

## Evaluate

Evaluation uses utterances only, without template inference or NER. CLINC's
training annotations are separate from evaluation. Download its pinned official
plus split and evaluate the merged encoder:

```bash
mkdir -p artifacts/public-evaluation
curl --fail --location \
  https://raw.githubusercontent.com/clinc/oos-eval/828f8093932c8fe6ca7936c3d2e52903b1c523de/data/data_oos_plus.json \
  --output artifacts/public-evaluation/data_oos_plus.json
.venv-legacy/bin/python scripts/evaluate_recovered.py \
  --checkpoint artifacts/public-training/runs/merged \
  --data-root data/massive \
  --clinc-splits-json artifacts/public-evaluation/data_oos_plus.json \
  --output artifacts/public-evaluation/results.json
```

The evaluator uses cosine nearest-neighbour search over training utterances,
HWU raw-file matching and CLINC out-of-scope examples, and records input hashes.

## Replaying augmentation from source

The public dataset is ready for training. To regenerate it from source, use
`scripts/prepare_table2_training.py`. Its settings are SNIPS top-2 with all
intents retained, ATIS top-2, MASSIVE and HWU top-3, and CLINC top-5.

```bash
PYTHONHASHSEED=42 .venv/bin/python scripts/prepare_table2_training.py \
  --output-dir artifacts/public-augmentation \
  --data-root data/massive \
  --clinc-splits-json artifacts/public-evaluation/data_oos_plus.json \
  --snips-top-k 2 --snips-retain-all-intents
```

This path uses supplied slot annotations for the annotated datasets and cached
spaCy annotations for CLINC. Preserve the manifest and compare the combined CSV
hash with the published dataset verification metadata. Source overlap and
duplicate records are retained; source dataset terms and citations apply.
