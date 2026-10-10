#!/usr/bin/env bash
#SBATCH --job-name=tadse-snips-top3
#SBATCH --array=0-2
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
[[ "$INDEX" =~ ^[0-2]$ ]] || exit 2
SEED=$((42 + INDEX))
DATA="$EXPERIMENT/training-data"
OUT="$EXPERIMENT/runs/seed-$SEED"
[[ ! -e "$OUT" ]] || { echo "Preserving existing run: $OUT" >&2; exit 2; }
cd "$CODE"
PY="$WORK/.venv/bin/python"
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=42
export HF_HOME="$WORK/artifacts/hf" PYTHONUNBUFFERED=1
"$PY" - "$DATA" "$WORK/artifacts/snips-top2-20261009/training-data/manifest.json" <<'PY'
import hashlib, json, sys
from pathlib import Path
data = Path(sys.argv[1])
m = json.loads((data/'manifest.json').read_text())
old = json.loads(Path(sys.argv[2]).read_text())
assert m['input_integrity_verified']
assert m['source_order'] == ['snips','atis','massive','hwu','clinc']
assert m['sources']['snips']['top_k'] == 3
assert m['sources']['snips']['trainable_retention'] == 'all_intents'
assert m['sources']['snips']['retained_rows'] == 440594
assert m['sources']['snips']['rows_per_output_open'] == [50770,188997,40437,39975,67786,16606,36023]
assert m['combined']['rows'] == 1111086
assert hashlib.sha256((data/'combined_train.csv').read_bytes()).hexdigest() == m['combined']['sha256']
for source in ('atis','massive','hwu','clinc'):
    assert m['sources'][source]['trainable']['sha256'] == old['sources'][source]['trainable']['sha256']
PY
mkdir -p "$OUT"
git rev-parse HEAD > "$OUT/code_commit.txt"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$OUT/gpu.csv"
"$PY" scripts/reproduce_train.py \
    --train-file "$DATA/combined_train.csv" \
    --model-name-or-path princeton-nlp/unsup-simcse-bert-base-uncased \
    --revision 218baf6bc48c463a53f5b4e1c568299537fab124 \
    --output-dir "$OUT/model" --max-steps -1 --num-train-epochs 2 \
    --batch-size 16 --learning-rate 1e-8 --max-length 32 --seed "$SEED" \
    --pooler-type cls --legacy-head-initialization --keep-last-batch \
    --simcse-utt-loss-ratio 1 --simcse-ptn-loss-ratio 1 \
    --pairwise-loss-ratio 0.5 --pairwise-neg-type utterance \
    --logging-steps 200 --save-steps 5000
"$PY" scripts/export_table2_head.py --checkpoint "$OUT/model" --output-dir "$OUT/head-export/model"
"$PY" scripts/evaluate_recovered.py --checkpoint "$OUT/head-export/model" \
    --data-root "$WORK/data/massive" \
    --clinc-splits-json "$WORK/artifacts/sources/data_oos_plus.json" \
    --output "$OUT/recovered-evaluation.json"
