# Table 2 headline-model replication

Scope: train **TaDSE** and **TaDSE w/ MLP**, with an initial SimCSE control. The
other DSE/TOD-BERT rows are not part of this first headline-model run. No dataset
or checkpoint upload, or GitHub push, is performed by this workflow.

## Recovered recipe

The [paper](https://arxiv.org/html/2305.14299v3#S2.T2) and Appendix C specify all
five augmented sources, SimCSE BERT-base initialization, two epochs, batch 16,
learning rate 1e-8, utterance/template/pairwise weights 1/1/0.5, and utterance
negatives for the pairwise objective. The eight-epoch exception applies to
single-source pattern-MLP experiments, not the pooled Table 2 runs.

| Model | SNIPS | ATIS | MASSIVE | HWU64 | CLINC | Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TaDSE target | 97.00 | 89.70 | 78.18 | 82.77 | 70.56 | 83.64 |
| TaDSE w/ MLP target | 96.29 | 89.14 | 79.15 | 82.29 | 72.49 | 83.87 |

Recovered defaults: training length 32; three temperatures 0.05; CLS/custom-MLP
training pooling; no new special slot token; inference through BERT's pooler at
length 128. No semantic compression is used for this comparison. Seed 42 is a
recorded reconstruction choice, not a recovered launch setting.

The historical initializer did not copy exported `pooler.dense` parameters into
the custom contrastive `mlp.dense`. The Table 2 launcher explicitly uses
`--legacy-head-initialization`; the earlier reconstruction's mapping remains
available for other runs. Missing custom heads are logged as newly initialized,
while missing encoder weights still fail. The MLP variant enables only
`apply_ptn_transform`, the separate template projection.

The original BERT forward/loss functions are AST-identical to private commit
`48d5f94`. The modern stock Trainer replaces historical private Trainer APIs;
PyTorch/Transformers versions, effective arguments and weight-loading details
are recorded. FP32, the stock optimizer/scheduler, and retained final incomplete
batches are explicit. Original launch arguments and checkpoints remain absent.

## Original augmentation, preserved

`legacy/augment_data.py` is a byte-for-byte recovery from the private iCloud
project, last changed in August 2023. Its SHA256 is
`d7f3daa57b58e79c391d86caa9e5cb1aebc8b98f44188dabef2307a59a1636a6`.
Its original behavior, including SNIPS truncation on each intent, is preserved.

`prepare_table2_training.py` runs its original `main` with output-write capture.
This saves both raw legacy bytes and a properly quoted/headered training CSV.
Each original write remains one logical record, even when it contains embedded
newlines. Opening the raw output with `w` resets the captured CSV too: it does
not restore the overwritten SNIPS intents. CLINC reuses its historical NER log
and original helper functions with the original default-empty slot book.
The literal `[ country ]` example is not invented into a new NER entity.

| Source | Retained training records | Raw file lines |
| --- | ---: | ---: |
| SNIPS | 159,099 | 163,261 |
| ATIS | 238,847 | 238,847 |
| MASSIVE | 78,289 | 78,289 |
| HWU64 | 133,176 | 133,176 |
| CLINC | 220,180 | 220,180 |
| Combined | 829,591 | 833,753 |

The raw counts round to the Table 1 figures. Physical lines and training records
are deliberately not conflated. Historical merging/header insertion is unknown;
the valid-CSV ingestion adapter is therefore an explicit difference, not proof
of byte-identical historical training input. `PYTHONHASHSEED=42` fixes legacy set
iteration for this replay. No new NER, extra split filtering, caps or value
normalization are applied. Source files and generation hashes are verified.

## Evaluation protocol and known split differences

Cosine 1NN uses every original reference example and every test example, with
first-reference tie handling. Overlaps are reported, not silently excluded.
There is no test-based checkpoint or compression selection.

| Source | References | Test | Source/protocol |
| --- | ---: | ---: | --- |
| SNIPS | 13,784 | 700 | Recovered `snips_full`: full JSON references, validation JSON used as test; original Latin-1 decoding |
| ATIS | 4,978 | 893 | Bundled Rasa JSON references/test, including combined intent labels |
| MASSIVE | 11,514 | 2,974 | Original train/test partitions |
| HWU64 | 9,960 | 1,076 | Pinned canonical KFold_1 labeled split, 64 intents |
| CLINC | 15,250 | 5,500 | Official plus split, OOS included |

HWU is a material protocol uncertainty. The recovered raw-matching path produces
only 830 test examples with the bundled files. It is evaluated separately as a
diagnostic, never substituted into the 1,076-example headline comparison.
The canonical labeled files and Git revisions are saved under
`artifacts/table2-sources/`. No missing labels are guessed.

Initial BERT-pooler control (job 54201): SNIPS **92.00**, ATIS **85.6663**,
MASSIVE **76.7653**, HWU **80.5762**, CLINC **71.00**. ATIS, MASSIVE and CLINC
match Table 2's SimCSE row to its printed precision. SNIPS and HWU differ, so
this is not a claim that every historical evaluation choice is recovered.

## Execution

Primary PW checkout: `/home/minsik/Template-Contrastive-Embedding`.
Run root: `artifacts/table2-20261007/`.

```bash
PYTHONHASHSEED=42 .venv/bin/python scripts/prepare_table2_training.py \
  --data-root data/massive \
  --clinc-splits-json artifacts/sources/data_oos_plus.json \
  --output-dir artifacts/table2-20261007/training-data-v2
```

Existing directories are preserved; choose new output paths for replays. Full
training uses an immutable Git checkout, one GPU per model (global batch 16),
periodic recovery checkpoints, final save/reload verification and automatic
five-dataset evaluation. `scripts/table2_pw.sbatch` takes code checkout, primary
checkout, training-data directory, benchmark JSON, fresh run directory and
`tadse` or `tadse_mlp` as its six arguments. It records the code commit and GPU.

Both variants use the same frozen corpus and benchmark manifest. Final results
are written to `<run>/table2.json`, with per-intent accuracy, wrong-prediction
identities, source/checkpoint hashes and deltas to the selected Table 2 row.
Completion of a job does not by itself establish matching the published scores.
