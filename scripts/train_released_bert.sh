#!/usr/bin/env bash
# Reproduce the two released BERT parents with the recovered custom CLTrainer.
set -euo pipefail
[[ $# == 3 ]] || { echo 'Usage: train_released_bert.sh DATA_CSV INITIAL_MODEL OUTPUT_DIR' >&2; exit 2; }
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$REPO/.venv-legacy/bin/python"
SENTEVAL="$REPO/artifacts/legacy-recovery-20261011/SentEval"
[[ -x "$PY" && -d "$SENTEVAL/senteval" ]] || { echo 'Run scripts/setup_recovered_training.sh first.' >&2; exit 2; }
DATA="$("$PY" -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "$1")"
MODEL="$("$PY" -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "$2")"
OUT="$("$PY" -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "$3")"
[[ ! -e "$OUT" ]] || { echo "Preserving existing output: $OUT" >&2; exit 2; }
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
"$PY" - "$DATA" "$MODEL" <<'PY'
import hashlib,sys
from pathlib import Path
import torch
assert torch.cuda.device_count() <= 1, "Select one GPU with CUDA_VISIBLE_DEVICES"
path,model=map(Path,sys.argv[1:])
assert hashlib.sha256(path.read_bytes()).hexdigest() == '9b8404042073814afbf0a63670e8ef9919d722cf8bee1b7aea208ecdad696dd4', 'Training CSV differs from release'
assert all((model/name).is_file() for name in ('config.json','pytorch_model.bin','vocab.txt')), 'Initial model incomplete'
PY
export PYTHONPATH="$SENTEVAL" OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=42
export PYTHONUNBUFFERED=1
cd "$REPO"
mkdir -p "$OUT"
for VARIANT in nonmlp mlp; do
  RUN="$OUT/$VARIANT"
  FLAGS=(--pooler_type cls)
  [[ "$VARIANT" != mlp ]] || FLAGS+=(--apply_ptn_transform)
  "$PY" scripts/run_simcse_head_ablation.py --audit "$RUN/initialization-audit.json" -- \
    --model_name_or_path "$MODEL" --train_file "$DATA" \
    --output_dir "$RUN/model" --dataset_cache_path "$RUN/dataset-cache" \
    --do_train --num_train_epochs 2 --per_device_train_batch_size 16 \
    --learning_rate 1e-8 --max_seq_length 32 --seed 42 \
    --simcse_utt_loss_ratio 1 --simcse_ptn_loss_ratio 1 --pairwise_loss_ratio 0.5 \
    --pairwise_neg_type utterance --logging_steps 200 --save_steps 5000 --save_total_limit 2 \
    "${FLAGS[@]}"
  EXPORT_FLAGS=(--checkpoint "$RUN/model")
  [[ "$VARIANT" != mlp ]] || EXPORT_FLAGS+=(--apply-ptn-transform)
  "$PY" scripts/export_recovered_training.py \
    --output-dir "$RUN/export" "${EXPORT_FLAGS[@]}"
done
"$PY" scripts/merge_exported_models.py --left "$OUT/nonmlp/export" \
  --right "$OUT/mlp/export" --output-dir "$OUT/merged"
