#!/bin/bash
#SBATCH --job-name=assemble_modis
#SBATCH --output=./log/slurm-%j.log
#SBATCH --error=./log/slurm-%j.err
#SBATCH --partition=scicore
#SBATCH --time=1-00:00:00
#SBATCH --qos=1day
# SBATCH --time=00:30:00
# SBATCH --qos=30min
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

# Activate conda environment
eval "$(/scicore/home/meiera/schulz0022/miniforge-pypy3/bin/conda shell.bash hook)"
conda activate gnt

# Change to project directory
cd /scicore/home/meiera/schulz0022/projects/growth-and-temperature

# Set DATA_NOBACKUP environment variable
export DATA_NOBACKUP="/scicore/home/meiera/schulz0022/projects/growth-and-temperature/data_nobackup"

# Calculate memory limit (leave some buffer for system - 60% of allocated)
MEMORY_LIMIT_GB=$(echo "scale=0; $SLURM_MEM_PER_NODE * 0.9 / 1024" | bc)

# Run the assembly using the unified interface
python run.py assemble \
    --config orchestration/configs/data.yaml \
    --source modis \
    --dask-threads $SLURM_CPUS_PER_TASK \
    --dask-memory-limit "${MEMORY_LIMIT_GB}GiB" \
    --temp-dir "${DATA_NOBACKUP}/scratch/dask" \
    --dashboard-port 8787 \
    --debug
