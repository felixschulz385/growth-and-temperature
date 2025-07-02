#!/bin/bash
#SBATCH --job-name=misc-preprocess
#SBATCH --output=./log/slurm-%j.out
#SBATCH --error=./log/slurm-%j.err
#SBATCH --time=06:00:00
#SBATCH --qos=6hours
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G

# Activate conda environment
eval "$(/scicore/home/meiera/schulz0022/miniforge-pypy3/bin/conda shell.bash hook)"
conda activate gnt

# Run the debugpy job
/scicore/home/meiera/schulz0022/miniforge-pypy3/envs/gnt/bin/python "/scicore/home/meiera/schulz0022/projects/growth-and-temperature/run.py" preprocess \
    --config "/scicore/home/meiera/schulz0022/projects/growth-and-temperature/orchestration/configs/data.yaml" \
    --source misc \
    --stage rasterize \
    --debug