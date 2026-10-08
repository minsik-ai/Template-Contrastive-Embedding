#!/usr/bin/env bash
#SBATCH --job-name=tadse-table2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --partition=voltagepark
#SBATCH --qos=normal
#SBATCH --account=yejinchoi
set -euo pipefail

CODE="${1:?immutable code checkout}"
WORK="${2:?primary checkout with venv and artifacts}"
DATA="${3:?verified legacy replay directory}"
BENCHMARKS="${4:?five-dataset benchmark JSON}"
OUT="${5:?fresh run directory}"
VARIANT="${6:?tadse or tadse_mlp}"
[[ "$VARIANT" == tadse || "$VARIANT" == tadse_mlp ]] || exit 2
[[ ! -e "$OUT" ]] || { echo "Preserving existing run: $OUT" >&2; exit 2; }
cd "$CODE"
PY="$WORK/.venv/bin/python"
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=42
export HF_HOME="$WORK/artifacts/hf" PYTHONUNBUFFERED=1
"$PY" - "$DATA" "$BENCHMARKS" <<'PY'
import hashlib, json, sys
from pathlib import Path
data, benchmarks = map(Path, sys.argv[1:])
manifest = json.loads((data / 'manifest.json').read_text())
assert manifest['input_integrity_verified'] is True
assert manifest['source_order'] == ['snips', 'atis', 'massive', 'hwu', 'clinc']
assert hashlib.sha256((data / 'combined_train.csv').read_bytes()).hexdigest() == manifest['combined']['sha256']
actual = json.loads(benchmarks.read_text())['tasks']
for name, n in {'snips':700,'atis':893,'massive':2974,'hwu':1076,'clinc':5500}.items():
    assert len(actual[name]['splits']['test']) == n, name
print('Verified five-source training corpus and all published test sizes.')
PY
mkdir -p "$OUT"
git rev-parse HEAD > "$OUT/code_commit.txt"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$OUT/gpu.csv"
FLAGS=()
[[ "$VARIANT" != tadse_mlp ]] || FLAGS+=(--apply-ptn-transform)
"$PY" scripts/reproduce_train.py \
    --train-file "$DATA/combined_train.csv" \
    --model-name-or-path princeton-nlp/unsup-simcse-bert-base-uncased \
    --revision 218baf6bc48c463a53f5b4e1c568299537fab124 \
    --output-dir "$OUT/model" --max-steps -1 --num-train-epochs 2 \
    --batch-size 16 --learning-rate 1e-8 --max-length 32 --seed 42 \
    --pooler-type cls --legacy-head-initialization --keep-last-batch \
    --simcse-utt-loss-ratio 1 --simcse-ptn-loss-ratio 1 \
    --pairwise-loss-ratio 0.5 --pairwise-neg-type utterance \
    --logging-steps 200 --save-steps 5000 "${FLAGS[@]}"
"$PY" scripts/evaluate_table2.py --benchmarks "$BENCHMARKS" \
    --checkpoint "$OUT/model" --output "$OUT/table2.json" \
    --device cuda --pooling bert-pooler --max-length 128 --paper-row "$VARIANT"
RAW_HWU="$(dirname "$BENCHMARKS")/benchmarks-raw-hwu.json"
if [[ -f "$RAW_HWU" ]]; then
    "$PY" scripts/evaluate_table2.py --benchmarks "$RAW_HWU" \
        --checkpoint "$OUT/model" --output "$OUT/hwu-raw-diagnostic.json" \
        --device cuda --pooling bert-pooler --max-length 128 --paper-row "$VARIANT" \
        --tasks hwu --allow-count-mismatch
fi
