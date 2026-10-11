# Original training stack recovery

The original entry point, custom CLTrainer, model, collator and declared
requirements are preserved under `legacy/training/` from private commit48d5f94.
Every preserved file is hash-verified. No original source edits were required.

## Runtime

A separate `.venv-legacy` on PW uses Python3.8.20, Transformers4.2.1,
datasets2.0.0, pandas1.1.5 and scikit-learn0.22.2. The original requirements do
not pin Torch. Torch2.2.2+cu121 was selected for H200 execution; it is not claimed
to be the unknown historical Torch version. NumPy1.21.6, PyArrow6.0.1 and
huggingface-hub0.10.1 provide compatible runtime dependencies. Full resolved
versions are stored with the experiment, separately from the recovered source.

The original trainer unconditionally imports SentEval. A real upstream checkout
at5d6dc117d1ac30357ddaeb7347fb5684e9b87015 is supplied through PYTHONPATH.
This is the latest upstream commit before the recovered code date, not a
recovered historical SentEval lock. No STS evaluation is requested; no fake
SentEval module or invented evaluation dataset is used.

`scripts/setup_recovered_training.sh` creates this isolated environment. It
installs the executable subset of original requirements and necessary runtime
dependencies; optional unrelated application packages are not installed.

## Completed smoke checks

Job57377 stopped at argument parsing because our launcher supplied the modern
`--report_to none` flag. Removing that unsupported launcher flag was the only
correction; original source stayed unchanged.

Job57386 completed both full-size BERT variants on40 training rows for two
epochs (six optimizer steps each), including component-loss logging, epoch-end
callbacks, periodic checkpoint saving/rotation and final saving. Exit0,45seconds.
Four exporter tests passed inside the legacy environment, including actual tiny
recovered-model reload and exact trained-head export in both variants.

The Python3.8 compatibility change in `evaluate_recovered.py` only rewrites a
parenthesized context-manager statement; data selection and scoring are unchanged.
The legacy-compatible exporter loads the original model class and maps the trained
utterance head into a standalone BERT pooler. It checks exact CPU reload outputs,
unchanged non-pooler encoder tensors and unchanged source files.

## Full replay

`scripts/recovered_training_pw.sbatch` runs the original script directly for
TaDSE and TaDSE with pattern MLP, seed42, two epochs, batch16, learning rate1e-8,
max length32, weights1/1/0.5 and utterance negatives. The legacy corpus remains
829591 rows with its verified original replay hash. The original CLTrainer
controls sampling, optimization, training, logging and saving; no stock-Trainer
replacement is used. Inference/export runs in the same legacy environment.
The SimCSE control from the legacy-environment export check is reused to expose
any runtime scoring difference without repeating that evaluation.

The historical launch command, merged CSV and full environment lock have not
been found, including a search of hidden project-local IDE/log files. Thus the
explicit paper settings and existing quoted legacy corpus remain reconstruction
inputs. Restoring executable code does not silently resolve those missing artifacts.
Existing checkpoints and the private Hub candidate remain untouched; GitHub is
not pushed.

The full-size export check initially exceeded the PW workspace container's2GB
memory limit (cgroup recorded an OOM kill). It was moved to a32GB Slurm allocation,
job57390. Full training jobs use32GB too. This did not require source changes.
