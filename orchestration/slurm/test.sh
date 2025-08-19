#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=./log/slurm-%j.out
#SBATCH --error=./log/slurm-%j.err
#SBATCH --partition=bigmem
#SBATCH --time=00:30:00
#SBATCH --qos=30min
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G

# Activate conda environment
eval "$(/scicore/home/meiera/schulz0022/miniforge-pypy3/bin/conda shell.bash hook)"
conda activate gnt

BODO_NUM_WORKERS=8
BODO_DISTRIBUTED_DIAGNOSTICS=1

# Calculate memory limit (leave some buffer for system - 60% of allocated)
MEMORY_LIMIT_GB=$(echo "scale=0; $SLURM_MEM_PER_NODE * 0.9 / 1024" | bc)

# Run with Dask settings from SLURM environment
mpiexec -n 8 /scicore/home/meiera/schulz0022/miniforge-pypy3/envs/gnt/bin/python /scicore/home/meiera/schulz0022/projects/growth-and-temperature/scripts/test.py