#!/bin/bash
#SBATCH --job-name=eog-preprocess
#SBATCH --output=./log/preprocess/viirs_annual/%x-%j.out
#SBATCH --error=./log/preprocess/viirs_annual/%x-%j.err
#SBATCH --time=06:00:00
#SBATCH --qos=6hours
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

mkdir -p "./log/preprocess/viirs_annual"

# Activate conda environment
eval "$(/scicore/home/meiera/schulz0022/miniforge-pypy3/bin/conda shell.bash hook)"
conda activate src

# Run the debugpy job
/scicore/home/meiera/schulz0022/miniforge-pypy3/envs/src/bin/python "/scicore/home/meiera/schulz0022/projects/growth-and-temperature/run.py" preprocess \
    --config "/scicore/home/meiera/schulz0022/projects/growth-and-temperature/orchestration/configs/data.yaml" \
    --source viirs_annual \
    --stage annual \
    --debug