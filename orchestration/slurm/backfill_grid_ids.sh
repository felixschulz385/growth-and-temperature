#!/bin/bash
#SBATCH --job-name=backfill-grid-ids
#SBATCH --output=./log/scripts/backfill_grid_ids/%x-%j.out
#SBATCH --error=./log/scripts/backfill_grid_ids/%x-%j.err
#SBATCH --qos=1day
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G

# Activate conda environment
eval "$(/scicore/home/meiera/schulz0022/miniforge-pypy3/bin/conda shell.bash hook)"
conda activate gnt

# Change to project directory
cd /scicore/home/meiera/schulz0022/projects/growth-and-temperature

/scicore/home/meiera/schulz0022/miniforge-pypy3/envs/gnt/bin/python -u "/scicore/home/meiera/schulz0022/projects/growth-and-temperature/scripts/backfill_grid_ids.py" \
  "$@"
