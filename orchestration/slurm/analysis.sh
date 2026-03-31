#!/bin/bash
#SBATCH --job-name=analysis
#SBATCH --partition=scicore
#SBATCH --time=0-12:00:00
#SBATCH --qos=1day
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G

# Usage: sbatch analysis.sh <model_name> [dataset_path]
MODEL=${1:?Usage: sbatch analysis.sh <model_name> [dataset_path]}

export WD="/scicore/home/meiera/schulz0022/projects/growth-and-temperature"

eval "$(/scicore/home/meiera/schulz0022/miniforge-pypy3/bin/conda shell.bash hook)"
conda activate gnt

# Determine duckreg version for structured log paths
DUCKREG_VERSION=$(python -c "from duckreg._version import __version__; print(__version__)" 2>/dev/null || echo "unknown")

LOG_DIR="${WD}/log/analysis/${MODEL}/${DUCKREG_VERSION}"
mkdir -p "${LOG_DIR}"

# Redirect all subsequent output to the model/version-specific log
exec > "${LOG_DIR}/slurm-${SLURM_JOB_ID}.log" 2> "${LOG_DIR}/slurm-${SLURM_JOB_ID}.err"

cd "${WD}"
mkdir -p "${WD}/scratch_nobackup/${SLURM_JOB_ID}"

python run.py analysis \
    --config orchestration/configs/analysis.xlsx \
    --model "${MODEL}" \
    --output output/analysis \
    ${DATASET:+--dataset "${DATASET}"}

echo "Done: ${MODEL} (duckreg ${DUCKREG_VERSION}) — results in ${WD}/output/analysis"
rm -Rf "${WD}/scratch_nobackup/${SLURM_JOB_ID}"