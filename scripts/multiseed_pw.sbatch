#!/usr/bin/env bash
#SBATCH --job-name=tadse-multiseed
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --partition=voltagepark
#SBATCH --qos=yejinchoi
#SBATCH --account=yejinchoi
set -euo pipefail
CODE="${1:?immutable code checkout}"
WORK="${2:?primary checkout}"
EXPERIMENT="${3:?experiment directory}"
INDEX="${SLURM_ARRAY_TASK_ID:?array index}"
[[ "$INDEX" =~ ^[0-3]$ ]] || exit 2
SEED=$((43 + INDEX / 2))
CONDITION=legacy_top5
DATA="$WORK/artifacts/table2-20261007/training-data-v2"
if (( INDEX % 2 )); then
  CONDITION=all_intent_top2
  DATA="$WORK/artifacts/snips-top2-20261009/training-data"
fi
BENCHMARKS="$WORK/artifacts/table2-20261007/benchmarks.json"
OUT="$EXPERIMENT/runs/$CONDITION/seed-$SEED"
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=42
export HF_HOME="$WORK/artifacts/hf" PYTHONUNBUFFERED=1
"$WORK/.venv/bin/python" - "$EXPERIMENT/manifest.json" "$DATA" "$BENCHMARKS" "$CONDITION" <<'PY'
import hashlib, json, sys
from pathlib import Path
manifest, data, benchmarks, condition = sys.argv[1:]
m = json.loads(Path(manifest).read_text())
assert hashlib.sha256((Path(data)/'combined_train.csv').read_bytes()).hexdigest() == m['training_sha256'][condition]
assert hashlib.sha256(Path(benchmarks).read_bytes()).hexdigest() == m['benchmark_sha256']
PY
bash "$CODE/scripts/table2_pw.sbatch" "$CODE" "$WORK" "$DATA" "$BENCHMARKS" "$OUT" tadse "$SEED"
cd "$CODE"
"$WORK/.venv/bin/python" scripts/probe_head_export.py \
  --checkpoint "$OUT/model" --benchmarks "$BENCHMARKS" \
  --original-test-results "$OUT/table2.json" --output-dir "$OUT/head-export" \
  --paper-row tadse --device cuda
"$WORK/.venv/bin/python" scripts/summarize_multiseed.py \
  --manifest "$EXPERIMENT/manifest.json" --output "$OUT/multiseed-summary.json"
