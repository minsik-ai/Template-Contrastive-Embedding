# Recovered evaluation is authoritative

As of 2026-10-10, use the recovered historical evaluation code and dataset
selection for every model comparison and release decision. Do not substitute a
canonical split to match a published row count. The original methodology stays
unchanged, including its preprocessing and reference selection.

In particular, the recovered HWU raw-file matching produces 830 test examples
with the available inputs. The previous 1,076-example canonical KFold1 results
remain historical reconstruction results and must not be mixed with recovered
scores or used to select the next release. Actual evaluated counts must be shown;
recovered-code execution alone does not prove that all historical data artifacts
or launch arguments have been recovered.

The four running seed jobs (55339 array0–3) retain their immutable training code.
Their old evaluation stages are historical outputs only. A subsequent recovered
code evaluation scores those exports, and existing seed42 exports are evaluated
with the same recovered protocol. No retraining or changes to augmentation,
losses, sampling or weights are required for this correction.

The exported trained head remains the chosen model representation; compare the
same representation across all candidates. Preserve existing Hub revisions until
a new candidate and its recovered-protocol results are selected explicitly.

## Utterance-only execution

Table2 evaluates utterances only. CLINC spaCy annotations are training inputs,
not evaluation requirements. The restored sources remain byte-identical. The
adapter executes recovered dataset-loader statements for the annotated datasets,
loads CLINC's original utterance/intent rows directly from the pinned official
plus JSON, and uses recovered SimCSE encoding and cosine nearest-neighbour search.
It bypasses the old shared script's unused template-loading/encoding path; it does
not fabricate CLINC test annotations or enable semantic compression.

All six non-MLP candidates (two recipes, seeds42/43/44) receive this same scoring
path. `scripts/recovered_eval_pw.sbatch` supports immediate existing-checkpoint
scoring and a dependency job for the active training array. Existing seed42 is
the original run; no checkpoint trained with numeric seed0 has been created.
