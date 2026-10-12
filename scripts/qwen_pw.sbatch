#!/usr/bin/env bash
#SBATCH --job-name=tadse-qwen-top2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --partition=voltagepark
#SBATCH --qos=yejinchoi
#SBATCH --account=yejinchoi
set -euo pipefail
CODE="${1:?immutable code checkout}"
WORK="${2:?primary checkout}"
EXP="${3:?experiment root}"
MODEL="$WORK/artifacts/hf/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
DATA="$WORK/artifacts/snips-top2-20261009/training-data/combined_train.csv"
OUT="$EXP/tadse-nonmlp-seed42"
PY="$WORK/.venv/bin/python"
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=42
export HF_HOME="$WORK/artifacts/hf" PYTHONUNBUFFERED=1
cd "$CODE"
"$PY" - "$DATA" "$EXP" <<'PY'
import hashlib,json,sys
from pathlib import Path
data,exp=map(Path,sys.argv[1:])
assert hashlib.sha256(data.read_bytes()).hexdigest()=='9b8404042073814afbf0a63670e8ef9919d722cf8bee1b7aea208ecdad696dd4'
assert json.loads((exp/'baseline.json').read_text())['complete']
v=json.loads((exp/'smoke/verification.json').read_text())
assert v['steps']==24 and v['export_max_abs_diff']==0
PY
"$PY" scripts/train_qwen.py --checkpoint "$MODEL" --train-file "$DATA" \
 --output-dir "$OUT" --legacy-transformers-source "$WORK/.venv-legacy/lib/python3.8/site-packages/transformers" \
 --seed 42 --batch-size 16 --epochs 2 --learning-rate 1e-8 --max-length 32 \
 --attention-dropout 0.1 --save-steps 5000 --logging-steps 200
"$PY" scripts/evaluate_qwen.py --checkpoint "$OUT/encoder" --data-root "$WORK/data/massive" \
 --clinc-splits-json "$WORK/artifacts/sources/data_oos_plus.json" --output "$OUT/recovered-evaluation.json" \
 --dimensions 768 --batch-size 16 --max-length 128
