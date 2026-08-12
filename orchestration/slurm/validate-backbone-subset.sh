#!/bin/bash
#SBATCH --job-name=validate-backbone-subset
#SBATCH --output=./log/maintenance/backbone_validate/%x-%j.out
#SBATCH --error=./log/maintenance/backbone_validate/%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --qos=6hours
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

# Runs scripts/validate_backbone_subset.py -- the backbone raster-pipeline
# validation from docs/design/05-migration.md step 8 (canonical grid ->
# kernel registry -> mask-aware convolution -> disc-ladder store ->
# tabularization on a small subset). See that script's docstring for what it
# does and does NOT check (it does not run the econometric sanity check
# against expected coefficient behaviour -- that needs src/analysis/ wiring).
#
# Usage:
#   sbatch orchestration/slurm/validate-backbone-subset.sh
#       -> synthetic mode: no real data needed, just confirms the pinned
#          "src" environment (environment.yml) runs the pipeline on this node.
#
#   sbatch orchestration/slurm/validate-backbone-subset.sh \
#       /path/to/eog_ease6933.zarr DNB_BRDF_Corrected_NTL "" "2020 2021"
#       -> real mode, once `run.py data run --source eog --step grid` has
#          been run with orchestration/configs/data.yaml's `pipeline.grid`
#          set to `ease6933` (it defaults to `legacy_4326`). The validity
#          mask is derived automatically from that store's own nodata/
#          _FillValue attribute -- leave the 3rd arg ("") empty unless the
#          store actually has a separate boolean mask variable to name.

PROJECT_ROOT="/scicore/home/meiera/schulz0022/projects/growth-and-temperature"
PYTHON_BIN="/scicore/home/meiera/schulz0022/miniforge-pypy3/envs/src/bin/python"

cd "$PROJECT_ROOT"
mkdir -p "./log/maintenance/backbone_validate"

LOG_FILE="./log/maintenance/backbone_validate/validate-backbone-subset-${SLURM_JOB_ID}.log"
exec > >(tee -a "$LOG_FILE")
exec 2>&1

# Activate conda environment
eval "$(/scicore/home/meiera/schulz0022/miniforge-pypy3/bin/conda shell.bash hook)"
conda activate src

VARIABLE_ZARR="${1:-}"
DATA_VAR="${2:-value}"
MASK_VAR="${3:-}"
YEARS="${4:-2020}"
HPC_ROOT="${HPC_ROOT:-${DATA_NOBACKUP:-$PROJECT_ROOT/data_nobackup}}"

echo "$(date -Is): Starting backbone subset validation (docs/design/05-migration.md step 8)"
echo "HPC root:      $HPC_ROOT"
echo "Variable zarr: ${VARIABLE_ZARR:-<none -- synthetic mode>}"
echo "Mask variable: ${MASK_VAR:-<none -- derived from nodata/_FillValue>}"
echo "Years:         $YEARS"
echo "Job ID:        $SLURM_JOB_ID"
echo "CPUs:          $SLURM_CPUS_PER_TASK"
echo "Memory:        ${SLURM_MEM_PER_NODE}MB"
echo "Log file:      $LOG_FILE"

ARGS=(--hpc-root "$HPC_ROOT" --years $YEARS)
if [ -n "$VARIABLE_ZARR" ]; then
    ARGS+=(--variable-zarr "$VARIABLE_ZARR" --data-var "$DATA_VAR")
    if [ -n "$MASK_VAR" ]; then
        ARGS+=(--mask-var "$MASK_VAR")
    fi
fi

"$PYTHON_BIN" "$PROJECT_ROOT/scripts/validate_backbone_subset.py" "${ARGS[@]}"

EXIT_CODE=$?
echo "$(date -Is): Backbone subset validation completed with exit code: $EXIT_CODE"
exit $EXIT_CODE
