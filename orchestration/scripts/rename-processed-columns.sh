#!/bin/bash
#SBATCH --job-name=rename-processed-columns
#SBATCH --output=./log/maintenance/rename-processed-columns/%x-%j.out
#SBATCH --error=./log/maintenance/rename-processed-columns/%x-%j.err
#SBATCH --time=06:00:00
#SBATCH --qos=6hours
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

# Usage: sbatch rename-processed-columns.sh [EXECUTE] [DATA_ROOT]
#   EXECUTE:   "--execute" to actually rewrite files; anything else (or
#              omitted) runs scripts/rename_processed_columns.py in its
#              default dry-run mode, which only logs what it WOULD rename.
#   DATA_ROOT: defaults to PROJECT_ROOT/data_nobackup.
#
# Renames, tile-by-tile, in already-materialized parquet:
#   - MODIS `valid_period_count_annual`/`valid_month_count_annual`
#     -> `valid_period_count_night_annual`/`valid_month_count_night_annual`
#   - EOG VIIRS `viirs_annual` -> `viirs_annual_avg`
# across the PREPARE-stage `modis_lst_21a2`/`eog_viirs_annual` tile trees and
# every already-assembled `assembled/grid=*/shake=*` tree -- see
# scripts/rename_processed_columns.py's module docstring for exactly which
# paths and why. One process, one tile (one `ix=/iy=` directory's parquet
# files) at a time -- not a SLURM array, since ~200-300 small per-tile
# rewrites comfortably fit in one job (pass --task-index / --print-manifest
# to scripts/rename_processed_columns.py directly instead, if you'd rather
# parallelize this over a SLURM array).
#
# Dry run first, always: `sbatch rename-processed-columns.sh` with no args,
# review the log, THEN `sbatch rename-processed-columns.sh --execute` once
# you're happy with the planned renames.

PROJECT_ROOT="/scicore/home/meiera/schulz0022/projects/growth-and-temperature"

EXECUTE_FLAG="${1:-}"
DATA_ROOT="${2:-${PROJECT_ROOT}/data_nobackup}"

mkdir -p "${PROJECT_ROOT}/log/maintenance/rename-processed-columns"

LOG_FILE="${PROJECT_ROOT}/log/maintenance/rename-processed-columns/rename-processed-columns-${SLURM_JOB_ID}.log"
exec > >(tee -a "$LOG_FILE")
exec 2>&1

eval "$(/scicore/home/meiera/schulz0022/miniforge-pypy3/bin/conda shell.bash hook)"
conda activate gnt

cd "$PROJECT_ROOT"

echo "$(date): Starting processed-columns rename job"
echo "Data root: $DATA_ROOT"
echo "Execute: ${EXECUTE_FLAG:-<dry run>}"
echo "Job ID: $SLURM_JOB_ID"
echo "Log file: $LOG_FILE"

EXECUTE_ARGS=()
if [ "$EXECUTE_FLAG" = "--execute" ]; then
    EXECUTE_ARGS=(--execute)
fi

/scicore/home/meiera/schulz0022/miniforge-pypy3/envs/gnt/bin/python \
    "${PROJECT_ROOT}/scripts/rename_processed_columns.py" \
    --data-root "$DATA_ROOT" \
    --all \
    "${EXECUTE_ARGS[@]}"

EXIT_CODE=$?
echo "$(date): Processed-columns rename job completed with exit code: $EXIT_CODE"

exit $EXIT_CODE
