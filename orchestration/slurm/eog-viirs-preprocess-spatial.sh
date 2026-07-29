#!/bin/bash
#SBATCH --job-name=eog-viirs-preprocess-spatial
#SBATCH --output=./log/preprocess/eog_viirs/%x-%j.out
#SBATCH --error=./log/preprocess/eog_viirs/%x-%j.err
#SBATCH --partition=scicore
#SBATCH --time=06:00:00
#SBATCH --qos=6hours
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G

mkdir -p "./log/preprocess/eog_viirs"

# Activate conda environment
eval "$(/scicore/home/meiera/schulz0022/miniforge-pypy3/bin/conda shell.bash hook)"
conda activate src

# Calculate memory limit (leave some buffer for system - 60% of allocated)
MEMORY_LIMIT_GB=$(echo "scale=0; $SLURM_MEM_PER_NODE * 0.9 / 1024" | bc)

# Run with Dask settings from SLURM environment
/scicore/home/meiera/schulz0022/miniforge-pypy3/envs/src/bin/python "/scicore/home/meiera/schulz0022/projects/growth-and-temperature/run.py" preprocess \
    --config "/scicore/home/meiera/schulz0022/projects/growth-and-temperature/orchestration/configs/data.yaml" \
    --source eog_viirs \
    --stage spatial \
    --dask-threads $SLURM_CPUS_PER_TASK \
    --dask-memory-limit "${MEMORY_LIMIT_GB}GiB" \
    --temp-dir "/scratch/schulz0022/glass_${SLURM_JOB_ID}" \
    --dashboard-port 8787 \
    --debug