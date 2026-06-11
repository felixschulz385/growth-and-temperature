#!/bin/bash
#SBATCH --job-name=migrate-result-paths
#SBATCH --output=./log/scripts/migrate_result_paths/%x-%j.out
#SBATCH --error=./log/scripts/migrate_result_paths/%x-%j.err
#SBATCH --qos=1day
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G

# Activate conda environment
eval "$(/scicore/home/meiera/schulz0022/miniforge-pypy3/bin/conda shell.bash hook)"
conda activate gnt

# Change to project directory
cd /scicore/home/meiera/schulz0022/projects/growth-and-temperature

/scicore/home/meiera/schulz0022/miniforge-pypy3/envs/gnt/bin/python -u "/scicore/home/meiera/schulz0022/projects/growth-and-temperature/scripts/migrate_result_paths.py" \
  "$@"
