#!/bin/bash
#SBATCH --job-name=analysis
#SBATCH --output=./log/slurm-%j.log
#SBATCH --error=./log/slurm-%j.err
#SBATCH --partition=scicore
#SBATCH --time=0-04:00:00
#SBATCH --qos=6hours
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

# Usage: sbatch analysis_modis_basic.sh <specification_name>
SPECIFICATION=${1:-modis_ntlharm_pooled}

# Activate conda environment
eval "$(/scicore/home/meiera/schulz0022/miniforge-pypy3/bin/conda shell.bash hook)"
conda activate gnt

# Change to project directory
cd /scicore/home/meiera/schulz0022/projects/growth-and-temperature

# Run the analysis using the unified interface
python run.py analysis \
    --config orchestration/configs/analysis.yaml \
    --analysis-type online_rls \
    --specification "${SPECIFICATION}" \
    --output output/analysis

echo "Analysis completed. Results saved to ${DATA_NOBACKUP}/analysis"
