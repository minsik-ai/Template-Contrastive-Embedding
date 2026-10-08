# Frozen-checkpoint head-export diagnostic

This tests one fixed hypothesis: serve the **trained contrastive MLP** as the
BERT pooler, as in the [official SimCSE conversion](https://github.com/princeton-nlp/SimCSE/blob/main/simcse_to_huggingface.py).
It does not retrain models or change encoder weights, tokenization, reference
sets, nearest-neighbor rules, or test populations.

## Conversion and verification

`scripts/export_table2_head.py` makes an independent copy of the trained BERT
encoder and maps only these tensors:

```text
mlp.dense.weight -> pooler.dense.weight
mlp.dense.bias   -> pooler.dense.bias
```

The output is a standalone `BertModel` with safetensors weights and tokenizer,
loadable with `AutoModel.from_pretrained`. Every other encoder tensor must remain
byte-identical. Reloaded pooler outputs are checked against the source's trained
MLP applied to CLS, using fixed probes with padding and truncation. Source files
and the original in-memory model are hashed before/after. Existing exports and
outputs inside the source checkpoint are refused.

## Diagnostic protocol

`scripts/probe_head_export.py` first verifies that the previous evaluation's
checkpoint/benchmark hashes and tokenization match this comparison. It then:

1. Performs the export and exact CPU reload checks.
2. Compares original and exported heads on MASSIVE validation (2,033 queries)
   and CLINC validation (3,100 queries), using full original references.
3. Evaluates the **same exported head on every test task**, regardless of the
   validation outcome. The original test scores are paired from the hash-verified
   existing reports.

The recorded benchmark has no separate validation split for SNIPS, ATIS or
canonical HWU. No substitute validation labels are invented. Historical overlaps
remain as documented in the benchmark; this probe does not claim new clean splits.
No per-dataset head selection or checkpoint selection is performed. Test results
are diagnostic comparisons, not evidence that the original paper used this export.

```bash
.venv/bin/python scripts/probe_head_export.py \
  --checkpoint artifacts/table2-20261007/runs/tadse/model \
  --benchmarks artifacts/table2-20261007/benchmarks.json \
  --original-test-results artifacts/table2-20261007/runs/tadse/table2.json \
  --output-dir artifacts/head-export-20261008/tadse \
  --paper-row tadse --device cuda
```

Repeat with `tadse_mlp` for the other frozen checkpoint. Use fresh output paths
for repeated probes. Per variant, `model/export-audit.json` stores the export
checks, `validation.json` records the first stage, and `results.json` stores the
complete comparison. Source checkpoints remain under `table2-20261007/runs/`.

The first diagnostic submission (54669, QoS `yejinchoi`) was replaced while
pending by **54670**, using the account-authorized `jit` QoS with a 15-minute
allocation. `jit` permits short jobs up to one hour on `voltagepark`; the full
two-hour training jobs appropriately remain on `yejinchoi` QoS. No Hub upload or
GitHub push is part of this diagnostic.

## Completed results — 2026-10-08

Job 54670 completed successfully in 1 minute 54 seconds. Both actual exports
passed source-file immutability and byte-identical encoder checks, with CPU
reload maximum absolute difference **0.0**.

Accuracy percentages below use the same fixed exported head for every task.

| Dataset | TaDSE original | TaDSE exported | Pattern-MLP original | Pattern-MLP exported |
| --- | ---: | ---: | ---: | ---: |
| SNIPS | 96.86 | 97.57 | 94.86 | 95.57 |
| ATIS | 89.14 | 89.59 | 87.46 | 87.68 |
| MASSIVE | 77.57 | 77.64 | 78.24 | 78.61 |
| HWU64 | 81.97 | 81.97 | 82.53 | 81.78 |
| CLINC | 69.55 | 69.45 | 71.02 | 70.56 |
| Mean | 83.02 | 83.24 | 82.82 | 82.84 |

Validation changes were positive in all four comparisons: TaDSE MASSIVE +0.05
and CLINC +0.32 points; pattern-MLP MASSIVE +0.25 and CLINC +0.55. Those gains
did not uniformly transfer to test scores. For the pattern-MLP model, gains on
SNIPS/ATIS/MASSIVE were largely offset by worse HWU/CLINC results.

This supports a small benefit for TaDSE but does not explain the full Table 2
gap, particularly for the MLP variant. Paper means remain 83.64 and 83.87.
It does not prove which export path the original experiments used. Encoder
weights, training, and the original serving checkpoints were not changed.

The standalone exports are on PW under:

```text
/home/minsik/Template-Contrastive-Embedding/artifacts/head-export-20261008/tadse/model
/home/minsik/Template-Contrastive-Embedding/artifacts/head-export-20261008/tadse_mlp/model
```

Use `AutoModel.from_pretrained(path).eval()` and **`pooler_output`** to obtain the
converted representation. No model or dataset was uploaded. Compact numeric
evidence is in [head-export-results.json](head-export-results.json); full
per-example validation/test results remain in the run artifact directories.
