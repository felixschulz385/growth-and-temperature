#!/bin/bash
# MODIS LST stage "annual": stream + composite via Planetary Computer's STAC
# catalog, then push the result to scicore over SSH.
#
# Deliberately NOT under orchestration/slurm/: unlike every other source's
# annual stage, this one needs outbound internet egress to Planetary
# Computer, which scicore's SLURM compute nodes may not have (unverified --
# docs/design/07-modis-ingest.md §2). Run this on whatever host does have
# that access -- a scicore login/transfer node, a workstation, a cloud VM --
# then it pushes results to scicore via `preprocess transfer`
# (docs/design/08-hpc-transfer.md) before orchestration/slurm/modis-
# preprocess-spatial.sh runs there as an ordinary SLURM job.
#
# If compute-node internet access is later confirmed, this can be adapted
# into a conventional SLURM script (mirroring
# orchestration/slurm/glass-modis-preprocess-annual.sh) with the transfer
# step simply skipped.
#
# Usage: modis-ingest-annual.sh --source modis [--year-range START END] [...]
#        (any extra flags are forwarded to `run.py preprocess run`)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

CONFIG="${MODIS_CONFIG:-$REPO_ROOT/orchestration/configs/data.yaml}"
SOURCE="modis"

mkdir -p "$REPO_ROOT/log/preprocess/modis"
cd "$REPO_ROOT"

# Allow --source to be overridden (e.g. modis_robustness_11a1) while
# forwarding every other flag through to `preprocess run` unchanged.
ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --source)
            SOURCE="$2"
            shift 2
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done

echo "Running MODIS stage annual (source=$SOURCE)"
python run.py preprocess run \
    --config "$CONFIG" \
    --source "$SOURCE" \
    --stage annual \
    "${ARGS[@]}"

echo "Stage annual complete -- pushing results to scicore"
python run.py preprocess transfer \
    --config "$CONFIG" \
    --source "$SOURCE" \
    --stage annual \
    --direction push
