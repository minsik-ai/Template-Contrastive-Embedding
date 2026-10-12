#!/usr/bin/env bash
#SBATCH --job-name=tadse-original
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --partition=voltagepark
#SBATCH --qos=yejinchoi
#SBATCH --account=yejinchoi
set -euo pipefail
CODE="${1:?immutable code checkout}"
WORK="${2:?primary checkout}"
EXP="${3:?experiment root}"
INITIALIZATION="${4:-original}"
[[ "$INITIALIZATION" == original || "$INITIALIZATION" == simcse-head ]] || exit 2
INDEX="${SLURM_ARRAY_TASK_ID:?array index}"
[[ "$INDEX" =~ ^[01]$ ]] || exit 2
VARIANT=tadse
[[ "$INDEX" == 0 ]] || VARIANT=tadse_mlp
OUT="$EXP/runs/$VARIANT-seed42"
[[ "$INITIALIZATION" == original ]] || OUT="$OUT-simcse-head"
[[ ! -e "$OUT" ]] || { echo "Preserving existing run: $OUT" >&2; exit 2; }
PY="$WORK/.venv-legacy/bin/python"
SENTEVAL="$WORK/artifacts/legacy-recovery-20261011/SentEval"
export PYTHONPATH="$SENTEVAL" OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=42
export HF_HOME="$WORK/artifacts/hf" PYTHONUNBUFFERED=1
DATA="$WORK/artifacts/table2-20261007/training-data-v2/combined_train.csv"
MODEL="$WORK/artifacts/hf/hub/models--princeton-nlp--unsup-simcse-bert-base-uncased/snapshots/218baf6bc48c463a53f5b4e1c568299537fab124"
cd "$CODE"
"$PY" - "$CODE" "$DATA" <<'PY'
import hashlib,json,sys
from pathlib import Path
code,data=map(Path,sys.argv[1:])
for entry in json.loads((code/'legacy/training/provenance.json').read_text())['files']:
 assert hashlib.sha256((code/'legacy/training'/entry['path']).read_bytes()).hexdigest()==entry['sha256']
assert hashlib.sha256(data.read_bytes()).hexdigest()=='817a281e1836b8100b115444017eb897490c1981696c1cca66f9eb061983e9c5'
PY
mkdir -p "$OUT"
git rev-parse HEAD > "$OUT/code_commit.txt"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$OUT/gpu.csv"
"$PY" - "$CODE" "$OUT" "$MODEL" "$VARIANT" <<'PY'
import dataclasses,json,sys
from pathlib import Path
code,out,model,variant=sys.argv[1:]
sys.path.insert(0,str(Path(code)/'scripts'))
from export_recovered_training import recovered_module
args=recovered_module().ModelArguments(model_name_or_path=model,pooler_type='cls',
 simcse_utt_loss_ratio=1.0,simcse_ptn_loss_ratio=1.0,pairwise_loss_ratio=0.5,
 pairwise_neg_type='utterance',apply_ptn_transform=variant=='tadse_mlp')
(Path(out)/'model_args.json').write_text(json.dumps(dataclasses.asdict(args),indent=2)+'\n')
PY
FLAGS=()
[[ "$VARIANT" != tadse_mlp ]] || FLAGS+=(--apply_ptn_transform)
COMMAND=("$PY" legacy/training/model/pairwise_pattern_train.py)
if [[ "$INITIALIZATION" == simcse-head ]]; then
 COMMAND=("$PY" scripts/run_simcse_head_ablation.py --audit "$OUT/initialization-audit.json" --)
fi
"${COMMAND[@]}" \
 --model_name_or_path "$MODEL" --train_file "$DATA" \
 --output_dir "$OUT/model" --dataset_cache_path "$OUT/dataset-cache" \
 --do_train --num_train_epochs 2 --per_device_train_batch_size 16 \
 --learning_rate 1e-8 --max_seq_length 32 --seed 42 --pooler_type cls \
 --simcse_utt_loss_ratio 1 --simcse_ptn_loss_ratio 1 --pairwise_loss_ratio 0.5 \
 --pairwise_neg_type utterance --logging_steps 200 --save_steps 5000 --save_total_limit 2 \
 "${FLAGS[@]}"
"$PY" scripts/export_recovered_training.py --checkpoint "$OUT/model" \
 --output-dir "$OUT/head-export/model" --model-args-json "$OUT/model_args.json"
"$PY" scripts/evaluate_recovered.py --checkpoint "$OUT/head-export/model" \
 --data-root "$WORK/data/massive" --clinc-splits-json "$WORK/artifacts/sources/data_oos_plus.json" \
 --output "$OUT/recovered-evaluation.json"
