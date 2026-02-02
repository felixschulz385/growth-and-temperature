#!/bin/bash
#SBATCH --job-name=analysis
#SBATCH --output=./log/analysis/slurm-%j.log
#SBATCH --error=./log/analysis/slurm-%j.err
#SBATCH --partition=scicore
#SBATCH --time=0-06:00:00
#SBATCH --qos=6hours
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

# Usage: sbatch analysis.sh <model_name> [dataset_path]
MODEL=${1}
DATASET=${2:-}  # Optional dataset path override

# Validate model name provided
if [ -z "$MODEL" ]; then
    echo "Error: Model name required"
    echo "Usage: sbatch analysis.sh <model_name> [dataset_path]"
    exit 1
fi

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
        --config orchestration/configs/analysis.xlsx \
        --model "${MODEL}" \
        --output output/analysis \
        --dataset "${DATASET}"
else
    python run.py analysis \
        --config orchestration/configs/analysis.xlsx \
        --model "${MODEL}" \
        --output output/analysis
fi

echo "Analysis completed for model: ${MODEL}"
echo "Results saved to ${WD}/output/analysis"

# Remove scratch dir
rm -Rf "${WD}/scratch_nobackup/${SLURM_JOB_ID}"