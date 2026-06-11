#!/bin/bash
#SBATCH --job-name=snl_mining-preprocess-spatial
#SBATCH --output=./log/preprocess/snl_mining/slurm-%j.log
#SBATCH --error=./log/preprocess/snl_mining/slurm-%j.err
#SBATCH --partition=scicore
#SBATCH --time=1-00:00:00
#SBATCH --qos=1day
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

# Activate conda environment
eval "$(/scicore/home/meiera/schulz0022/miniforge-pypy3/bin/conda shell.bash hook)"
conda activate gnt

# Change to project directory
cd /scicore/home/meiera/schulz0022/projects/growth-and-temperature

# Ensure log directory exists
mkdir -p ./log/preprocess/snl_mining

# Calculate memory limit (leave some buffer for system - 60% of allocated)
MEMORY_LIMIT_GB=$(echo "scale=0; $SLURM_MEM_PER_NODE * 0.6 / 1024" | bc)

# Run spatial preprocessing for the full inferred SNL mining year range.
# Omitting --year-range lets the preprocessor use all years available in the stage-0 DuckDB.
/scicore/home/meiera/schulz0022/miniforge-pypy3/envs/gnt/bin/python "/scicore/home/meiera/schulz0022/projects/growth-and-temperature/run.py" preprocess \
    --config "/scicore/home/meiera/schulz0022/projects/growth-and-temperature/orchestration/configs/data.yaml" \
    --source snl_mining \
    --stage spatial \
    --override \
    --dask-threads $SLURM_CPUS_PER_TASK \
    --dask-memory-limit "${MEMORY_LIMIT_GB}GiB" \
    --temp-dir "/scratch/schulz0022/snl_mining_${SLURM_JOB_ID}" \
    --dashboard-port 8787 \
    --debug
