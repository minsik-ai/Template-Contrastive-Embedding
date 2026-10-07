#!/usr/bin/env bash
#SBATCH --job-name=tadse-data-release
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
DATA="${2:?completed generation directory}"
RELEASE="${3:?new release directory}"
HF_REPO="${4:?private destination in authenticated user namespace}"
cd "$REPO"
PY="$REPO/.venv/bin/python"
"$PY" scripts/inspect_dataset_statistics.py \
    --manifest "$DATA/manifest.json" --data-root "$REPO/data/massive" \
    --clinc-splits-json "$REPO/artifacts/sources/data_oos_plus.json"
# Any failed Table 1 check exits 2 here, before the uploader can contact the Hub.
"$PY" scripts/prepare_dataset_release.py \
    --manifest "$DATA/release-manifest.json" --output-dir "$RELEASE"
"$PY" scripts/publish_private_dataset.py --release-dir "$RELEASE" --repo-id "$HF_REPO"
