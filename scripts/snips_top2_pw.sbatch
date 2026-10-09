#!/usr/bin/env bash
#SBATCH --job-name=tadse-snips-top2
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
WORK="${2:?primary checkout with artifacts and venv}"
DATA="${3:?verified top-2 corpus}"
BENCHMARKS="${4:?fixed benchmark JSON}"
OUT="${5:?fresh run directory}"
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=42
export HF_HOME="$WORK/artifacts/hf" PYTHONUNBUFFERED=1
"$WORK/.venv/bin/python" - "$DATA/manifest.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
s = m['sources']['snips']
assert s['top_k'] == 2 and s['retained_rows'] == 115378
assert s['rows_per_output_open'] == [15975, 28663, 13644, 13276, 22412, 8792, 12616]
assert m['combined']['rows'] == 785870
PY
bash "$CODE/scripts/table2_pw.sbatch" "$CODE" "$WORK" "$DATA" "$BENCHMARKS" "$OUT" tadse
cd "$CODE"
"$WORK/.venv/bin/python" scripts/probe_head_export.py \
  --checkpoint "$OUT/model" --benchmarks "$BENCHMARKS" \
  --original-test-results "$OUT/table2.json" --output-dir "$OUT/head-export" \
  --paper-row tadse --device cuda
