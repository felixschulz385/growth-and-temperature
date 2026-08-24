#!/bin/bash
#SBATCH --job-name=clean-prepare-scratch
#SBATCH --output=./log/maintenance/clean-prepare-scratch/%x-%j.out
#SBATCH --error=./log/maintenance/clean-prepare-scratch/%x-%j.err
#SBATCH --time=12:00:00
#SBATCH --qos=1day
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Usage: sbatch clean-prepare-scratch.sh [EXECUTE] [CONFIG_PATH] [SOURCE]
#   EXECUTE:     "--execute" to actually delete matched entries; anything
#                else (or omitted) runs scripts/clean_prepare_scratch.py in
#                its default dry-run mode, which only logs what it WOULD
#                delete.
#   CONFIG_PATH: defaults to orchestration/configs/data.yaml under PROJECT_ROOT.
#   SOURCE:      restrict the scan to one source id (default: every source
#                in the config).
#
# Dry run first, always: `sbatch clean-prepare-scratch.sh` with no args,
# review the log, THEN `sbatch clean-prepare-scratch.sh --execute` once
# you're happy with the planned deletions.

PROJECT_ROOT="/scicore/home/meiera/schulz0022/projects/growth-and-temperature"

EXECUTE_FLAG="${1:-}"
CONFIG_PATH="${2:-${PROJECT_ROOT}/orchestration/configs/data.yaml}"
SOURCE_ID="${3:-}"

mkdir -p "${PROJECT_ROOT}/log/maintenance/clean-prepare-scratch"

LOG_FILE="${PROJECT_ROOT}/log/maintenance/clean-prepare-scratch/clean-prepare-scratch-${SLURM_JOB_ID}.log"
exec > >(tee -a "$LOG_FILE")
exec 2>&1

eval "$(/scicore/home/meiera/schulz0022/miniforge-pypy3/bin/conda shell.bash hook)"
conda activate src

cd "$PROJECT_ROOT"

echo "$(date): Starting PREPARE-scratch cleanup job"
echo "Config: $CONFIG_PATH"
echo "Source: ${SOURCE_ID:-<all>}"
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

/scicore/home/meiera/schulz0022/miniforge-pypy3/envs/src/bin/python \
    "${PROJECT_ROOT}/scripts/clean_prepare_scratch.py" \
    --config "$CONFIG_PATH" \
    "${SOURCE_ARGS[@]}" \
    "${EXECUTE_ARGS[@]}"

EXIT_CODE=$?
echo "$(date): PREPARE-scratch cleanup job completed with exit code: $EXIT_CODE"

exit $EXIT_CODE
