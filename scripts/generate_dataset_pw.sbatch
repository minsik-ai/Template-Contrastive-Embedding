#!/usr/bin/env bash
#SBATCH --job-name=tadse-data-gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --partition=voltagepark
#SBATCH --qos=normal
#SBATCH --account=yejinchoi
set -euo pipefail

REPO="${1:?absolute checkout directory}"
OUTPUT="${2:?fresh generation directory}"
cd "$REPO"
# Preserve the existing generator and per-dataset k values. No new NER is run:
# four datasets provide annotations; CLINC uses its archived NER annotations.
"$REPO/.venv/bin/python" data/massive/augment_data.py \
    --output-dir "$OUTPUT" \
    --clinc-splits-json "$REPO/artifacts/sources/data_oos_plus.json"
