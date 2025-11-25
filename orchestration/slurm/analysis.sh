#!/bin/bash
#SBATCH --job-name=analysis
#SBATCH --output=./log/slurm-%j.log
#SBATCH --error=./log/slurm-%j.err
#SBATCH --partition=scicore
#SBATCH --time=0-06:00:00
#SBATCH --qos=6hours
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

# Usage: sbatch analysis.sh <type> <specification> [dataset_path]
TYPE=${1:-online_rls}
SPECIFICATION=${2:-modis_ntlharm_pooled}
DATASET=${3:-}  # Optional dataset path override

# Activate conda environment
eval "$(/scicore/home/meiera/schulz0022/miniforge-pypy3/bin/conda shell.bash hook)"
conda activate gnt

# Change to project directory and set WD environment variable
WD="/scicore/home/meiera/schulz0022/projects/growth-and-temperature"
cd "${WD}"
export WD  # Make WD available to the Python script for env var expansion

# Create scratch dir
mkdir -p "${WD}/scratch_nobackup/${SLURM_JOB_ID}"

# Run the analysis using the unified interface
if [ -n "$DATASET" ]; then
    python run.py analysis \
        --config orchestration/configs/analysis.yaml \
        --analysis-type "${TYPE}" \
        --specification "${SPECIFICATION}" \
        --output output/analysis \
        --dataset "${DATASET}"
else
    python run.py analysis \
        --config orchestration/configs/analysis.yaml \
        --analysis-type "${TYPE}" \
        --specification "${SPECIFICATION}" \
        --output output/analysis
fi

echo "Analysis completed. Results saved to ${WD}/output/analysis"

# Remove scratch dir
rm -Rf "${WD}/scratch_nobackup/${SLURM_JOB_ID}"