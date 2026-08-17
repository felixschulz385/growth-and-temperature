#!/bin/bash
#SBATCH --job-name=migrate-legacy-layout
#SBATCH --output=./log/maintenance/migrate-legacy-layout/%x-%j.out
#SBATCH --error=./log/maintenance/migrate-legacy-layout/%x-%j.err
#SBATCH --time=12:00:00
#SBATCH --qos=1day
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

# Usage: sbatch migrate-legacy-layout.sh [EXECUTE] [CONFIG_PATH] [GRID_ID]
#   EXECUTE:     "--execute" to actually move files; anything else (or
#                omitted) runs scripts/migrate_legacy_layout.py in its
#                default dry-run mode, which only logs what it WOULD move.
#   CONFIG_PATH: defaults to orchestration/configs/data.yaml under PROJECT_ROOT.
#   GRID_ID:     "legacy_4326" (default) or "ease6933" -- which grid's GRID-
#                stage stores to migrate. Ignored for modis/
#                modis_robustness_11a1, which always use ease6933.
#
# Dry run first, always: `sbatch migrate-legacy-layout.sh` with no args,
# review the log, THEN `sbatch migrate-legacy-layout.sh --execute` once
# you're happy with the planned moves.

PROJECT_ROOT="/scicore/home/meiera/schulz0022/projects/growth-and-temperature"

EXECUTE_FLAG="${1:-}"
CONFIG_PATH="${2:-${PROJECT_ROOT}/orchestration/configs/data.yaml}"
GRID_ID="${3:-legacy_4326}"

mkdir -p "${PROJECT_ROOT}/log/maintenance/migrate-legacy-layout"

LOG_FILE="${PROJECT_ROOT}/log/maintenance/migrate-legacy-layout/migrate-legacy-layout-${SLURM_JOB_ID}.log"
exec > >(tee -a "$LOG_FILE")
exec 2>&1

eval "$(/scicore/home/meiera/schulz0022/miniforge-pypy3/bin/conda shell.bash hook)"
conda activate src

cd "$PROJECT_ROOT"

echo "$(date): Starting legacy-layout migration job"
echo "Config: $CONFIG_PATH"
echo "Grid id: $GRID_ID"
echo "Execute: ${EXECUTE_FLAG:-<dry run>}"
echo "Job ID: $SLURM_JOB_ID"
echo "Log file: $LOG_FILE"

EXECUTE_ARGS=()
if [ "$EXECUTE_FLAG" = "--execute" ]; then
    EXECUTE_ARGS=(--execute)
fi

/scicore/home/meiera/schulz0022/miniforge-pypy3/envs/src/bin/python \
    "${PROJECT_ROOT}/scripts/migrate_legacy_layout.py" \
    --config "$CONFIG_PATH" \
    --grid-id "$GRID_ID" \
    "${EXECUTE_ARGS[@]}"

EXIT_CODE=$?
echo "$(date): legacy-layout migration job completed with exit code: $EXIT_CODE"

exit $EXIT_CODE
