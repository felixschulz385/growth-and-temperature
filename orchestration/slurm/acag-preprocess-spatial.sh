#!/bin/bash
#SBATCH --job-name=acag-preprocess-spatial
#SBATCH --output=./log/preprocess/acag/%x-%j.out
#SBATCH --error=./log/preprocess/acag/%x-%j.err
#SBATCH --time=1-00:00:00
#SBATCH --qos=1day
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G

mkdir -p "./log/preprocess/acag"

# Activate conda environment
eval "$(/scicore/home/meiera/schulz0022/miniforge-pypy3/bin/conda shell.bash hook)"
conda activate src

# Calculate memory limit (leave some buffer for system - 90% of allocated)
MEMORY_LIMIT_GB=$(echo "scale=0; $SLURM_MEM_PER_NODE * 0.6 / 1024" | bc)

# Run preprocessing for ESA CCI land cover (annual stage)
/scicore/home/meiera/schulz0022/miniforge-pypy3/envs/src/bin/python "/scicore/home/meiera/schulz0022/projects/growth-and-temperature/run.py" preprocess \
    --config "/scicore/home/meiera/schulz0022/projects/growth-and-temperature/orchestration/configs/data.yaml" \
    --source acag \
    --stage spatial \
    --dask-threads $SLURM_CPUS_PER_TASK \
    --dask-memory-limit "${MEMORY_LIMIT_GB}GiB" \
    --temp-dir "/scratch/schulz0022/esacci_${SLURM_JOB_ID}" \
    --dashboard-port 8787 \
    --debug
