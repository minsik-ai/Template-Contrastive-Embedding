> Current release (2026-10-12): the released BERT model uses the verified
> all-intent top-2 corpus, now published privately as
> [template-dialogue-pairs](https://huggingface.co/datasets/minsik-oh/template-dialogue-pairs).
> See the [dataset card](../releases/tadse-top2-dialogue-pairs/README.md) and
> [release manifest](../releases/tadse-top2-dialogue-pairs/manifest.json).
> The pre-recovery paper-count audit below is historical and remains unchanged.

# Private dataset release workflow

This report describes the **pre-recovery reconstructed candidate**. The original
2023 generator was subsequently found in iCloud and preserved in `legacy/`.
Current headline-model work uses that recovered source; see
[Table 2 replication](table2-replication.md). No private upload has occurred.

The requested destination is `minsik-oh/tadse-dialogue-augmentation-20261007`,
private. Generation is **not** permission to publish mismatched data: the user
requires verification against paper Table 1 before any upload. No GitHub push is
part of this workflow.

## Generate without changing the existing augmentation code

`scripts/generate_dataset_pw.sbatch` queues the existing pooled generation path.
Its initial 2026-10-07 audit detected count mismatches, including differences
introduced by pooled cross-source held-out filtering. To inspect each source
independently while preserving the generator itself:

```bash
.venv/bin/python scripts/generate_release_sources.py \
  --output-dir artifacts/release-20261007/source-isolated \
  --data-root data/massive \
  --clinc-splits-json artifacts/sources/data_oos_plus.json
```

Use a fresh output directory for every attempt. This command runs the unchanged
generator separately for SNIPS, ATIS, MASSIVE, HWU64 and CLINC, then combines the
outputs and their provenance. It retains default k values 5/2/3/3/5 and does not
cap or truncate data. Only CLINC consumes historical NER annotations; all other
sources use their supplied annotations. Per-source generation deliberately
retains cross-dataset overlaps, which must be audited separately before model
evaluation.

`inspect_dataset_statistics.py` records the actual original examples consumed by
each generation, including generic templates and typed slot/value annotations.
It verifies the loaded source paths and hashes against generation provenance.
Original manifests are preserved. No augmentation strategy is changed.

## Verify before packaging or uploading

```bash
.venv/bin/python scripts/prepare_dataset_release.py \
  --manifest artifacts/release-20261007/source-isolated/release-manifest.json \
  --output-dir artifacts/release-20261007/source-isolated-review
```

The gate streams every CSV and verifies its hash, declared row count, expected
k value, absence of truncation, original-example evidence, source input hashes,
all per-source Table 1 counts, totals and combined content/order. Counts printed
with K suffixes are checked using the displayed precision and an explicit
nearest-unit rounding assumption; the paper does not supply exact raw counts or
its rounding convention. Matching these intervals alone would not prove identical
historical samples or implementation.

Failure writes `verification.json` and a clearly blocked draft card, returns
exit code 2, and does not create Parquet files or contact Hugging Face. Success
packages one verified Parquet file per source and a dataset card with a combined
configuration plus individual source configurations.

## Private publication, only after the gate succeeds

```bash
.venv/bin/python scripts/publish_private_dataset.py \
  --release-dir artifacts/release-20261007/source-isolated-review \
  --repo-id minsik-oh/tadse-dialogue-augmentation-20261007
```

The uploader repeats the source/count audit and verifies packaged hashes before
any Hub API call. It creates a **new private dataset**, refuses an existing
repository, checks privacy before and after upload, and uploads only the five
Parquet files, card and verification report. It does not upload environments,
credentials, checkpoints or unrelated files. No public visibility option exists.

For the pooled path, `release_dataset_pw.sbatch` chains original-statistics
inspection, the verification gate and private publication using `set -e`, so a
failed audit prevents the uploader from running.

## Audit artifacts

- Initial generation: job 53803; pooled audit/publication gate: job 53815.
- Per-source generation and audit/publication gate: job 53818.
- PW root: `/home/minsik/Template-Contrastive-Embedding/artifacts/release-20261007/`.
- `private-release/verification.json`: pooled audit, 22 failed checks.
- `source-isolated-review/verification.json`: per-source audit after job 53818.

## Result, 2026-10-07

Generation completed, but publication was blocked. Job 53818 returned the
intentional verification exit code 2 after 83 of 98 checks passed. The uploader
was never reached. Hugging Face confirms the proposed dataset repository does
not exist; no data or draft card was uploaded.

| Source | Paper augmented utterances | Independently generated utterances |
| --- | ---: | ---: |
| SNIPS | 163K | 4,657,571 |
| ATIS | 239K | 237,710 |
| MASSIVE | 78K | 77,900 |
| HWU64 | 133K | 133,026 |
| CLINC | 220K | 220,122 |
| Total | 834K | 5,326,329 |

All integrity, provenance, top-k, uncapped-generation and combined-content checks
passed. The remaining 15 failures are published-count comparisons (including
per-source original/slot/template counts and totals). In particular, matching an
augmented-utterance figure does not mean all of that source's Table 1 statistics
match. The full reports remain on PW; a compact durable summary is in
[dataset-release-status.json](dataset-release-status.json).

Validation: 51 distinct tests passed on PW, including real Parquet packaging and
mocked private-upload boundaries. Ruff, Python compilation and shell syntax
checks passed. The existing augmentation source was not modified.

The source history contains no original SNIPS augmentation implementation to
recover: `augment_data.py` first appeared in the reconstruction commit. Do not
invent a sampling limit or alter k to force the 163K figure. The audit also
distinguishes ATIS's 79 role-qualified slot names from the paper's 41 categories;
stripping dotted role prefixes yields 41, but this is a diagnostic observation,
not an applied generator change. Some ATIS offsets disagree with their supplied
annotation values; these also require reconciliation before claiming fidelity.
Read-only diagnostics found 192 of 16,560 ATIS training entities whose offsets
match their annotation values after subtracting one; all other spans already
match. The annotated value strings number 647, whereas offset-derived strings
number 705 before filtering. Those findings were not used to silently change the
generator or bypass the publication check.
