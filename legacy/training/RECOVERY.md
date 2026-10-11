# Recovered training implementation

The six files listed in `provenance.json` are exact Git blobs from private TaDSE commit `48d5f94`. The entry point imports the adjacent `tadse` package, so this directory is isolated from the modern live `model/` implementation. No compatibility edits have been made to these source files. Python parsing and byte/hash equality against Git were checked; this source recovery alone does not demonstrate successful execution.

## Entry point

Use a compatible environment and run `python legacy/training/model/pairwise_pattern_train.py` with the original underscore-style arguments. A paper-parameter replay command (not a recovered historical launch command) is:

```sh
python legacy/training/model/pairwise_pattern_train.py \
  --model_name_or_path princeton-nlp/unsup-simcse-bert-base-uncased \
  --model_revision 218baf6bc48c463a53f5b4e1c568299537fab124 \
  --train_file /absolute/path/to/combined_train.csv \
  --output_dir /absolute/path/to/new-output-directory \
  --do_train --num_train_epochs 2 --per_device_train_batch_size 16 \
  --learning_rate 1e-8 --max_seq_length 32 --seed 42 \
  --pooler_type cls --simcse_utt_loss_ratio 1 \
  --simcse_ptn_loss_ratio 1 --pairwise_loss_ratio 0.5 \
  --pairwise_neg_type utterance --apply_ptn_transform false
```

The source's own pairwise defaults are **0.01 / pattern**, so launch overrides are material. The command above matches the paper-parameter interpretation used in the replication, not evidence of the historical invocation. With no evaluation requested, there is no STS best-checkpoint selection; the entry point saves the final model. Run smoke/import checks before a full job. The source initially loads a tokenizer with hardcoded `revision="main"` before loading it again with the requested revision; a local immutable pretrained snapshot avoids that initial network revision ambiguity, but the entry point then treats that directory as a possible resume source.

## Concrete behavioral differences from the modern runner

- **Training implementation:** this entry point constructs the original `CLTrainer`, including its own train loop, sampler, checkpoint method, and loss-component logging. The modern runner constructs stock Transformers `Trainer`. Both call the model's returned TaDSE loss, but that alone does not prove loop equivalence.
- **Sampler:** recovered `CLTrainer._get_train_sampler` returns plain `RandomSampler(dataset)` for one process and `DistributedSampler(dataset)` otherwise. The modern stack has separate `data_seed`, Accelerate preparation, and newer sampler behavior. Matching a seed does not establish identical sample order or dropout RNG consumption.
- **Optimizer/scheduler:** the original loop inherits `create_optimizer_and_scheduler` from its installed Transformers version. It clips gradients, steps optimizer then scheduler, and calls `model.zero_grad()`. The modern runner does not explicitly select an optimizer and therefore inherits the newer stack's default. Their exact instantiated implementations/settings must be recorded at runtime, not inferred from equal learning rates.
- **Input parser:** the original uses `datasets.load_dataset` and the first two columns. It preserves string whitespace and replaces `None` with one space. The modern `read_pairs` strips both strings, rejects blanks, and requires particular header names. The historical merged input CSV has not been recovered; the current quoted CSV is an adapter-produced artifact.
- **Four-view collation:** original preprocessing tokenizes two utterance copies and two template copies, then collates each example as `[utterance, utterance, pattern, pattern]`. The modern collator uses that same view order and dynamic padding for ordinary rows. The original additionally supports slot-token replacement and fixed-padding flags; those were not enabled in the replication.
- **Initialization:** original uses direct `BertForPatternCL.from_pretrained` with no custom pooler-to-MLP remapping and then resizes token embeddings. The replication's `--legacy-head-initialization` prevents its optional remapping, but initialization behavior can still depend on library version.
- **Checkpoints:** when evaluation metrics and `metric_for_best_model` are supplied, the recovered trainer saves only improving models at the output root. Otherwise it saves ordinary step checkpoints. Its `evaluate` runs SentEval STS tasks, not intent benchmarks. The modern replication selected its final model and later performed intent evaluation.

## Missing environment and artifact information

- `requirements.txt` pins Transformers 4.2.1, datasets 2.0.0, pandas 1.1.5 and other packages, but **does not pin Torch**, SciPy, or several other dependencies. This is not a fully resolved environment lock.
- `trainers.py` unconditionally imports `senteval` after prepending `/home/omsus20/PycharmProjects/TaDSE/senteval/`. Neither that package nor its STS data is tracked in this commit. A real compatible SentEval package must be supplied even for import-only/training-only execution; do not replace it with a fake module. STS data is needed if training evaluation is enabled.
- The exact original CLI, original packaged training CSV, training framework's fully resolved versions, and historical checkpoint-selection settings have not been recovered. No shell launch files or root README exist in the source commit.
- This recovery covers the contrastive training entry point and every adjacent local import it uses. The separate supervised classifier pipeline is outside this entry point's imports and was not copied.
