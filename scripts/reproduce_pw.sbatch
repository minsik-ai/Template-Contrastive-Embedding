#!/usr/bin/env bash
#SBATCH --job-name=tadse-reproduce
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --partition=voltagepark
#SBATCH --qos=normal
#SBATCH --account=yejinchoi
set -euo pipefail

# Supply absolute paths; Slurm runs a copy of this file outside the checkout.
REPO="${1:?absolute checkout directory}"
OUT="${2:?new output directory}"
TRAIN="${3:?training CSV}"
STEPS="${4:--1}"
cd "$REPO"
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$REPO/artifacts/hf}"
mkdir -p "$(dirname "$OUT")"
test ! -e "$OUT" || { echo "Output already exists: $OUT" >&2; exit 2; }
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
"$REPO/.venv/bin/python" scripts/reproduce_train.py \
    --train-file "$TRAIN" \
    --model-name-or-path princeton-nlp/unsup-simcse-bert-base-uncased \
    --output-dir "$OUT" --max-steps "$STEPS" --num-train-epochs 2 \
    --batch-size 16 --learning-rate 1e-8 --pairwise-loss-ratio 0.5 \
    --pairwise-neg-type utterance --seed 42
