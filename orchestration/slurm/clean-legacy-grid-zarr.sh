#!/bin/bash
#SBATCH --job-name=clean-legacy-grid-zarr
#SBATCH --output=./log/maintenance/clean-legacy-grid-zarr/%x-%j.out
#SBATCH --error=./log/maintenance/clean-legacy-grid-zarr/%x-%j.err
#SBATCH --time=12:00:00
#SBATCH --qos=1day
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Usage: sbatch clean-legacy-grid-zarr.sh [EXECUTE] [CONFIG_PATH] [SOURCE] [GRID_ID]
#   EXECUTE:     "--execute" to actually delete matched entries; anything
#                else (or omitted) runs scripts/clean_legacy_grid_zarr.py in
#                its default dry-run mode, which only logs what it WOULD
#                delete.
#   CONFIG_PATH: defaults to orchestration/configs/data.yaml under PROJECT_ROOT.
#   SOURCE:      restrict the scan to one source id (default: every pixel-grid
#                source in the config).
#   GRID_ID:     "legacy_4326" (default) or "ease6933" -- ignored for
#                modis/modis_robustness_11a1, which always check ease6933.
#
# Deletes leftover <family>.zarr GRID stores left behind by sources that have
# since switched to writing cell_id-keyed parquet parts instead
# (src.data.common.prepare.driver.run_tiled_prepare) -- see
# scripts/clean_legacy_grid_zarr.py's module docstring.
#
# Dry run first, always: `sbatch clean-legacy-grid-zarr.sh` with no args,
# review the log, THEN `sbatch clean-legacy-grid-zarr.sh --execute` once
# you're happy with the planned deletions.

PROJECT_ROOT="/scicore/home/meiera/schulz0022/projects/growth-and-temperature"

EXECUTE_FLAG="${1:-}"
CONFIG_PATH="${2:-${PROJECT_ROOT}/orchestration/configs/data.yaml}"
SOURCE_ID="${3:-}"
GRID_ID="${4:-}"

mkdir -p "${PROJECT_ROOT}/log/maintenance/clean-legacy-grid-zarr"

LOG_FILE="${PROJECT_ROOT}/log/maintenance/clean-legacy-grid-zarr/clean-legacy-grid-zarr-${SLURM_JOB_ID}.log"
exec > >(tee -a "$LOG_FILE")
exec 2>&1

eval "$(/scicore/home/meiera/schulz0022/miniforge-pypy3/bin/conda shell.bash hook)"
conda activate src

cd "$PROJECT_ROOT"

echo "$(date): Starting legacy GRID-zarr cleanup job"
echo "Config: $CONFIG_PATH"
echo "Source: ${SOURCE_ID:-<all>}"
echo "Grid id: ${GRID_ID:-<default: legacy_4326>}"
echo "Execute: ${EXECUTE_FLAG:-<dry run>}"
echo "Job ID: $SLURM_JOB_ID"
echo "Log file: $LOG_FILE"

EXECUTE_ARGS=()
if [ "$EXECUTE_FLAG" = "--execute" ]; then
    EXECUTE_ARGS=(--execute)
fi

SOURCE_ARGS=()
if [ -n "$SOURCE_ID" ]; then
    SOURCE_ARGS=(--source "$SOURCE_ID")
fi

GRID_ID_ARGS=()
if [ -n "$GRID_ID" ]; then
    GRID_ID_ARGS=(--grid-id "$GRID_ID")
fi

/scicore/home/meiera/schulz0022/miniforge-pypy3/envs/src/bin/python \
    "${PROJECT_ROOT}/scripts/clean_legacy_grid_zarr.py" \
    --config "$CONFIG_PATH" \
    "${SOURCE_ARGS[@]}" \
    "${GRID_ID_ARGS[@]}" \
    "${EXECUTE_ARGS[@]}"

EXIT_CODE=$?
echo "$(date): Legacy GRID-zarr cleanup job completed with exit code: $EXIT_CODE"

exit $EXIT_CODE
