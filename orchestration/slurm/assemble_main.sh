#!/bin/bash
#SBATCH --job-name=assemble_main
#SBATCH --output=./log/slurm-%j.out
#SBATCH --error=./log/slurm-%j.err
#SBATCH --partition=scicore
#SBATCH --time=1-00:00:00
#SBATCH --qos=1day
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Activate conda environment
eval "$(/scicore/home/meiera/schulz0022/miniforge-pypy3/bin/conda shell.bash hook)"
conda activate gnt

# Load necessary modules
module load Java/21.0.2

# Set environment variables for data paths
export DATA_NOBACKUP="/scicore/home/meiera/schulz0022/projects/growth-and-temperature/data_nobackup"

# Export SLURM variables for Spark configuration (optional, can be overridden by config)
export SPARK_CPUS=${SLURM_CPUS_PER_TASK:-32}
export SPARK_MEMORY_GB=$((${SLURM_MEM_PER_NODE:-262144} / 1024))

# Change to project directory
cd /scicore/home/meiera/schulz0022/projects/growth-and-temperature

# Run the assembly using the unified interface
python run.py assemble \
    --config orchestration/configs/data.yaml \
    --source main \
    --log-file "./log/assemble-main-${SLURM_JOB_ID}.log"