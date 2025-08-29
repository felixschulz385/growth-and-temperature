#!/bin/bash
#SBATCH --job-name=rechunk-annual-zarrs
#SBATCH --output=./log/rechunk-%j.out
#SBATCH --error=./log/rechunk-%j.err
#SBATCH --time=12:00:00
#SBATCH --qos=1day
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

CHUNK_SIZE="${1:-256}"
BASE_PATH="${2:-/scicore/home/meiera/schulz0022/projects/growth-and-temperature}"

# Create log directory if it doesn't exist
mkdir -p "${BASE_PATH}/log"

# Set up logging
LOG_FILE="${BASE_PATH}/log/rechunk-zarr-${SLURM_JOB_ID}.log"
exec > >(tee -a "$LOG_FILE")
exec 2>&1

# Activate conda environment
eval "$(/scicore/home/meiera/schulz0022/miniforge-pypy3/bin/conda shell.bash hook)"
conda activate gnt

echo "$(date): Starting batch zarr rechunking job"
echo "Base path: $BASE_PATH"
echo "Chunk size: $CHUNK_SIZE"
echo "Job ID: $SLURM_JOB_ID"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: ${SLURM_MEM_PER_NODE}MB"
echo "Log file: $LOG_FILE"

# Run the rechunking script for all annual zarr files
/scicore/home/meiera/schulz0022/miniforge-pypy3/envs/gnt/bin/python \
    "/scicore/home/meiera/schulz0022/projects/growth-and-temperature/scripts/rechunk_zarr.py" \
    --base-path "$BASE_PATH" \
    --chunk-size "$CHUNK_SIZE"

EXIT_CODE=$?
echo "$(date): Batch rechunking job completed with exit code: $EXIT_CODE"

exit $EXIT_CODE
