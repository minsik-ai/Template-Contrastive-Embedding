# All-intent SNIPS top-2 experiment

The release candidate remains the exported non-pattern-MLP TaDSE checkpoint
(mean 83.2443). Its private Hub revision and exact file checksums are recorded in
`releases/tadse-bert-base-uncased/manifest.json`. GitHub is not pushed.

This follow-up changes SNIPS top-k from 5 to 2 and retains all seven intents in
the trainable CSV, instead of only the final SearchScreeningEvent output.
The recovered generator itself remains unchanged. The adapter still preserves
its raw overwritten output separately; the trainable CSV explicitly accumulates
all complete write records with one header.

Expected SNIPS count: 115,378 pairs. Per-intent counts in generator order:
15,975 / 28,663 / 13,644 / 13,276 / 22,412 / 8,792 / 12,616.
Other source CSVs must have identical hashes to the baseline. Total: 785,870.
Counts include original utterances and per-example deduplication. No new NER,
filtering, sampling, global deduplication, or augmentation algorithm is introduced.
The complete top-k count sweep is in `snips-topk-counts.json`.

## Fixed comparison

Run only non-pattern-MLP TaDSE, initialized from the same pinned SimCSE revision,
seed 42, two epochs, batch 16, learning rate 1e-8, length 32, FP32 and identical
loss settings. The existing Table2 runner is reused. Two epochs now imply 98,234
optimizer steps rather than 103,700; this is an equal-epoch comparison, not an
equal-step or isolated top-k ablation. Intent coverage changes too.

`scripts/snips_top2_pw.sbatch` uses account `yejinchoi`, partition `voltagepark`,
QoS `yejinchoi`, one GPU and a two-hour allocation. It validates expected counts,
trains, evaluates the original pooler, then exports the trained head and evaluates
that fixed head on the same five benchmarks. Compare exported to exported.
One seed cannot establish robust improvements. Do not replace the private release
candidate automatically based on these exploratory test results.

```bash
PYTHONHASHSEED=42 .venv/bin/python scripts/prepare_table2_training.py \
  --output-dir artifacts/snips-top2-20261009/training-data \
  --data-root data/massive \
  --clinc-splits-json artifacts/sources/data_oos_plus.json \
  --snips-top-k 2 --snips-retain-all-intents
```
