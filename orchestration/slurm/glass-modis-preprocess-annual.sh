#!/bin/bash
#SBATCH --job-name=glass-modis-preprocess
#SBATCH --output=./log/slurm-%j.out
#SBATCH --error=./log/slurm-%j.err
#SBATCH --time=7-00:00:00
#SBATCH --qos=1week
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

# Activate conda environment
eval "$(/scicore/home/meiera/schulz0022/miniforge-pypy3/bin/conda shell.bash hook)"
conda activate gnt

# Calculate memory limit (leave some buffer for system - 90% of allocated)
MEMORY_LIMIT_GB=$(echo "scale=0; $SLURM_MEM_PER_NODE * 0.9 / 1024" | bc)

# Run with Dask settings from SLURM environment
/scicore/home/meiera/schulz0022/miniforge-pypy3/envs/gnt/bin/python "/scicore/home/meiera/schulz0022/projects/growth-and-temperature/run.py" preprocess \
    --config "/scicore/home/meiera/schulz0022/projects/growth-and-temperature/orchestration/configs/data.yaml" \
    --source glass_modis \
    --stage annual \
    --dask-threads $SLURM_CPUS_PER_TASK \
    --dask-memory-limit "${MEMORY_LIMIT_GB}GiB" \
    --temp-dir "/scratch/schulz0022/glass_${SLURM_JOB_ID}" \
    --dashboard-port 8787 \
    --debug