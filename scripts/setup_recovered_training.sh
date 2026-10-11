#!/usr/bin/env bash
# Isolated runnable subset of the recovered requirements; never edits original code.
set -euo pipefail
WORK="${1:?absolute primary repository path}"
cd "$WORK"
LEGACY_ENV="$WORK/.venv-legacy"
SENTEVAL="$WORK/artifacts/legacy-recovery-20261011/SentEval"
[[ -x "$LEGACY_ENV/bin/python" ]] || uv venv --python 3.8.20 "$LEGACY_ENV"
# Torch was unpinned in the recovered requirements; cu121 supports the PW H200.
uv pip install --python "$LEGACY_ENV/bin/python" \
  'transformers==4.2.1' 'datasets==2.0.0' 'pandas==1.1.5' \
  'scikit-learn==0.22.2' 'setuptools==49.3.0' 'prettytable==2.1.0' \
  'torch==2.2.2' 'numpy==1.21.6' 'pyarrow==6.0.1' 'huggingface-hub==0.10.1'
if [[ ! -d "$SENTEVAL/.git" ]]; then
  git clone https://github.com/facebookresearch/SentEval.git "$SENTEVAL"
  git -C "$SENTEVAL" checkout 5d6dc117d1ac30357ddaeb7347fb5684e9b87015
fi
[[ "$(git -C "$SENTEVAL" rev-parse HEAD)" == 5d6dc117d1ac30357ddaeb7347fb5684e9b87015 ]]
uv pip freeze --python "$LEGACY_ENV/bin/python" > "$WORK/artifacts/legacy-recovery-20261011/environment.freeze.txt"
PYTHONPATH="$SENTEVAL" "$LEGACY_ENV/bin/python" "$WORK/legacy/training/model/pairwise_pattern_train.py" --help > "$WORK/artifacts/legacy-recovery-20261011/help.txt"
