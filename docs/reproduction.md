# TaDSE experiment reconstruction

This is the earlier reconstruction log. The original augmentation source was
subsequently recovered from iCloud. See [Table 2 replication](table2-replication.md)
for the current headline experiments using the preserved 2023 generator.

This work reconstructs missing dependencies and synthetic data generation from
the released code and paper. It does **not** establish that the published scores
have been reproduced. Generated corpora, caches, environments, logs, and weights
live under ignored `artifacts/` or `.venv/`; only code and compact evidence belong
in Git.

## Sources and recovered recipe

- [Paper, arXiv v3](https://arxiv.org/html/2305.14299v3), especially Sections 3–4
  and Appendices A/C; [ACL paper](https://aclanthology.org/2026.acl-long.1015.pdf).
- `data/massive/extract_ptn_clinc.py`: Cartesian slot substitutions, top-k
  frequencies, original utterance retention and per-original-row deduplication.
- Bundled MASSIVE, SNIPS, ATIS, HWU64 inputs and historical CLINC annotation log.
- [Official CLINC splits](https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_oos_plus.json),
  SHA256 `bfcca9ae515623541dc1983c94c4ed7cae9d26b42ae47d74b972e51bb6f7a21f`.

For each slot type, count values in training examples, retain its top k, and
enumerate their Cartesian combinations within each template. Keep original
utterances and use generic `{SLOT}` templates. Preserve multiplicity across
different original rows. Paper k values: SNIPS 5, ATIS 2, MASSIVE 3, HWU64 3,
CLINC 5. This is deterministic slot substitution, not LLM paraphrase generation.

Appendix C specifies `princeton-nlp/unsup-simcse-bert-base-uncased`, learning rate
1e-8, batch size 16, two epochs (eight for single-source pattern-MLP experiments).
Set utterance/template/pairwise loss weights to 1/1/0.5 and pairwise negatives
to utterances. The original CLI defaults of 0.01 and pattern negatives disagree
with this recipe. The main comparison pools five sources; a MASSIVE-only run or
a short run is a partial reconstruction. Qwen3 is a supervised comparison, not
the principal TaDSE pretraining backbone in the recovered recipe.

## Cluster and environment

The new checkout is `/home/minsik/Template-Contrastive-Embedding` on PW, alongside
`/home/minsik/InferLLM-Repo`. It uses its own `.venv` and artifact directory.
See [dependencies.md](dependencies.md) for pins and legacy dependency gaps.

```bash
cd /home/minsik/Template-Contrastive-Embedding
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
uv pip check --python .venv/bin/python
mkdir -p artifacts/setup artifacts/sources artifacts/logs
uv pip freeze --python .venv/bin/python > artifacts/setup/environment.freeze.txt
curl --fail --location \
  https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_oos_plus.json \
  --output artifacts/sources/data_oos_plus.json
.venv/bin/python data/massive/augment_data.py \
  --output-dir artifacts/augmentation-clean \
  --clinc-splits-json artifacts/sources/data_oos_plus.json
```

The generator writes CSV headers and uses CSV quoting, unlike the legacy raw
comma-separated writes. Casefolded, whitespace-normalized held-out text is excluded from both original
training rows and generated utterances. The manifest records source/output
hashes, row counts, retained slot values, exclusions and repairs. Normalized filtering,
stable order and preserved punctuation are deliberate reconstruction choices.
An explicit `--max-combinations` produces a capped diagnostic corpus and must
not be represented as the complete augmentation.

## Verification and training

```bash
.venv/bin/python -m unittest discover -s tests -v
# Run on an allocated compute node, not a shared login node:
OMP_NUM_THREADS=2 TADSE_RUN_TRAINING_SMOKE=1 \
  .venv/bin/python -m unittest discover -s tests -v

# First run a bounded GPU pilot (100 optimizer steps):
sbatch --output=artifacts/logs/%x_%j.out scripts/reproduce_pw.sbatch \
  "$PWD" "$PWD/artifacts/runs/massive-pilot" \
  "$PWD/artifacts/augmentation-clean/massive_train.csv" 100
# Full two-epoch MASSIVE-only reconstruction: omit the final 100 argument.
# For the five-source experiment, pass combined_train.csv after reviewing gaps.
```

The wrapper defaults to a 20-minute allocation for the initial MASSIVE run.
For larger reviewed corpora, explicitly override Slurm `--time` based on the
measured steps/second and the expected number of optimizer steps.

The new runner reuses the existing BERT contrastive objective and its four views
`[utterance, utterance, pattern, pattern]` with the maintained Transformers
Trainer. It preserves the historical entrypoint for reference. Each run saves
model arguments, tokenizer, training arguments, resolved backbone revision,
input hash, metrics and checkpoint reload verification. Tiny random BERT checks
exercise the full forward/backward/save/reload path without downloading weights;
they are infrastructure tests, not meaningful learned models.

The official SimCSE exporter renames its MLP to `pooler.dense`. The new loader
copies those verified loaded weights into TaDSE's `mlp.dense` instead of silently
using a random projection. Missing encoder weights fail explicitly. It also
declares the existing shared LM-head bias alias so safetensors save/reload works.
The name mapping comes from the [official SimCSE conversion script](https://github.com/princeton-nlp/SimCSE/blob/main/simcse_to_huggingface.py).

## MASSIVE downstream check

```bash
.venv/bin/python scripts/evaluate_reproduction.py \
  --checkpoint artifacts/runs/massive-pilot \
  --massive-source data/massive/massive_en-US.jsonl \
  --output artifacts/results/massive-pilot.json --device cuda
# Repeat with --baseline and a new output filename for the initialized model.
```

The evaluator embeds all reference utterances/templates and selects compression
on validation from 0, 0.1, 0.2 and 0.5. It mixes `(1-lambda)*utterance +
lambda*template`, then normalizes for cosine 1NN. It checks the actual fine-tuning
CSV hash and excludes normalized training/query overlaps before any optional
sampling. Further reference/validation/test duplicates are removed and counted.
This cleaned protocol differs from unfiltered historical evaluation. The baseline
uses identical TaDSE inference settings; it is not labeled standard SimCSE's
CLS-only benchmark. No test-label-driven coefficient selection is performed.
Compression uses the dataset's supplied slot annotations for query templates;
these scores do not establish performance with automatically inferred slots.

Unrecovered settings are recorded explicitly rather than presented as paper facts:
seed 42, maximum length 32, CLS pooling with the recovered projection retained at
inference, no newly added slot token, stock AdamW/linear decay with no warmup,
and dropping an incomplete final training batch. Full effective Transformers
training arguments are saved in `run_args.json`. Pattern-MLP training is off for
the initial reconstruction.

## Generated on PW, 2026-10-06

All five sources were processed without a combinations cap into
`artifacts/augmentation-clean/`. Filtering uses the union of 16,804 normalized
held-out strings. Counts therefore differ from source-isolated generation.

| Source | Generated rows |
| --- | ---: |
| SNIPS | 4,657,559 |
| ATIS | 237,710 |
| MASSIVE | 73,317 |
| HWU64 | 111,472 |
| CLINC | 219,988 |
| Combined | 5,300,046 |

Combined SHA256:
`c9d0aae011f55a142c4167bb3b9f1aaf34b8cda863cb1d5661a0a6332f956d52`.
The full manifest includes every source hash, selected slot value and repair.
An earlier exact-string-only reconstruction remains separately preserved under
`artifacts/augmentation/`; use the normalized-filtered `augmentation-clean` data.

## Completed checks and queued experiment

- Python 3.11 environment installed on PW; all 64 resolved packages passed
  `uv pip check`.
- All 28 tests passed with the ML integration tests enabled, including a real
  tiny BERT optimizer update, exact save/reload, SimCSE projection mapping,
  rejection of missing encoder weights and cosine retrieval.
- Pretrained BERT GPU pilot (job 53236): 100 steps on the 73,317-row MASSIVE
  reconstruction, loss 1.168542, checkpoint reload maximum difference 0.0.
- Full-size MASSIVE downstream checks (job 53237): 11,514 references, 2,022
  validation queries and 2,948 test queries after filtering. Both models selected
  compression 0.5 on validation. No source queries matched the audited normalized
  synthetic training utterances.

| Model | Plain 1NN test accuracy | Validation-selected compression test accuracy |
| --- | ---: | ---: |
| Initialized TaDSE baseline | 76.66% | 79.34% |
| 100-step pilot | 76.73% | 79.27% |

The pilot does not demonstrate an improvement or reproduce the paper's training
budget. A full two-epoch MASSIVE-only train/evaluate job, **53238**, has been
submitted and is queued at this checkpoint of the work. It writes
`artifacts/runs/massive-2epochs/` and `artifacts/results/massive-2epochs.json`.
Its results are not yet included in this report. Check `squeue -j 53238` or
`sacct -j 53238` and `artifacts/logs/tadse-massive_53238.out` for live status.
Compact evidence is saved in [reproduction-results.json](reproduction-results.json).

## Missing parts and deviations

| Item | Evidence / action |
| --- | --- |
| Original augmentation code and dependency lock | Neither is present in Git history for the referenced filenames. Reconstructed and explicitly versioned here. |
| Original checkpoints, seed list and exact selected configurations | Not recovered; new weights cannot be called the original release. Seed 42, maximum length 32, pooling and other unrecovered choices must be identified as reconstruction defaults. |
| SNIPS size mismatch | Full bundled training plus top-5 independent Cartesian substitution produces millions of rows, rather than the paper's 163k. The smaller non-full files also fail to match. Do not silently cap or change k to fit a reported count. |
| CLINC source membership | The log's 15,250 records match official train + oos_train. Unfiltered top-5 yields 220,181 rows. Five upstream training texts overlap held-out splits; the clean reconstruction excludes overlap. Original spaCy version remains unknown. |
| ATIS offset corruption | A malformed sole span for `one way` exceeds text bounds. The generator narrowly repairs this recoverable full-text annotation and records it. Other invalid spans fail. |
| Split completeness | Bundled SNIPS has validation but no separately identified paper test; ATIS has no separate validation; HWU training membership is inferred by exclusion. |
| Historical trainer | Copied private Transformers internals and undefined evaluation paths make the old runner unreliable. New runner uses stock Trainer while preserving the existing objective. |
| RoBERTa and Qwen3 | RoBERTa forward references undefined variables/unsupported arguments; Qwen3 wraps a second backbone and uses BERT-specific forward arguments. New reproduction runner rejects these paths pending independent repair and tests. |
| Legacy evaluation | Spark utilities depend on missing `alexa_dr_semantic_similarity`; no validated five-dataset end-to-end paper evaluator exists. |
| Licensing / release | Source attribution is preserved. Dataset redistribution terms and original checkpoint provenance still require review before public Hub release. No Hub upload is performed here. |

Do not interpret a successful installation, a finite training loss, or a matching
checkpoint reload as reproduction of the paper's accuracy. Numerical replication
requires matching the corpus, hyperparameters and full downstream protocol.
