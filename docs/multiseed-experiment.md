> Protocol update (2026-10-10): running training jobs are preserved. Their
> reconstructed evaluation outputs remain historical diagnostics; rank models
> using subsequent recovered-code evaluations, consistently across all seeds.
> See `recovered-evaluation.md`.

# Paired non-MLP TaDSE replication across seeds

Compare existing legacy top-5 and all-intent top-2 training corpora at seeds
42, 43 and 44. Seed-42 runs are reused without selecting checkpoints. Four new
jobs run the other two matched seeds. Every run uses the trained-head export,
fixed final two-epoch checkpoint, full five-dataset benchmarks and no pattern MLP.

The frozen CSVs, source order, benchmark hashes, pretrained revision, model/loss
implementation and hyperparameters remain fixed. Only the training seed changes:
initialization of the custom head, dropout and training shuffle. PYTHONHASHSEED
remains 42 to preserve the data-generation convention. The runner's optional
seventh argument selects the training seed; its default remains 42.

Legacy: 829,591 records, SNIPS 159,099 from SearchScreeningEvent only.
Top-2: 785,870 records, SNIPS 115,378 across all seven intents.
Both train for two epochs, so optimizer steps differ (103,700 vs 98,234).
This compares two corpus recipes; it does not isolate top-k from intent coverage,
mixture weights or optimizer-step count.

`scripts/multiseed_pw.sbatch` maps array indices 0/1 to legacy/top2 seed43 and
2/3 to legacy/top2 seed44. Each job uses one GPU, account/QoS yejinchoi and a
two-hour allocation. It trains, evaluates, exports and evaluates again, then
writes a summary snapshot under its run directory. The last completed job's
snapshot includes all finished pairs. The summary can also be rerun explicitly:

```bash
python scripts/summarize_multiseed.py --manifest EXPERIMENT/manifest.json \
  --output EXPERIMENT/summary.json
```

Report per-task and five-task macro mean/sample standard deviation, plus paired
(top2 - legacy) differences across matching seeds. Missing pairs are explicit and
excluded together; incomplete output is not the final result. Three seeds give a
first estimate of variance, not strong evidence of significance. Averaging scores
does not improve a single checkpoint, and this experiment does not ensemble models
or automatically replace the private release candidate.
