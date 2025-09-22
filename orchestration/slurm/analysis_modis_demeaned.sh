#!/bin/bash
#SBATCH --job-name=analysis_modis_demeaned
#SBATCH --output=./log/slurm-%j.log
#SBATCH --error=./log/slurm-%j.err
#SBATCH --partition=scicore
#SBATCH --time=0-04:00:00
#SBATCH --qos=6hours
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

# Activate conda environment
eval "$(/scicore/home/meiera/schulz0022/miniforge-pypy3/bin/conda shell.bash hook)"
conda activate gnt

# Change to project directory
cd /scicore/home/meiera/schulz0022/projects/growth-and-temperature

# Set environment variables
export DATA_NOBACKUP="/scicore/home/meiera/schulz0022/projects/growth-and-temperature/data_nobackup"
export TMPDIR="${DATA_NOBACKUP}/scratch/tmp"

# Create scratch directory if it doesn't exist
mkdir -p "${TMPDIR}"

# Run the analysis using the unified interface
python run.py analysis \
    --config orchestration/configs/analysis.yaml \
    --analysis-type online_rls \
    --specification modis_demeaned \
    --output "${DATA_NOBACKUP}/output/analysis" \
    --debug

echo "Analysis completed. Results saved to ${DATA_NOBACKUP}/analysis"
