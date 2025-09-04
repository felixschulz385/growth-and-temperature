#!/bin/bash
#SBATCH --job-name=demean_modis
#SBATCH --output=./log/slurm-%j.log
#SBATCH --error=./log/slurm-%j.err
#SBATCH --partition=scicore
#SBATCH --time=06:00:00
#SBATCH --qos=6hours
# SBATCH --time=00:30:00
# SBATCH --qos=30min
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G

# Set environment variables for Spark
export SPARK_MEMORY_GB=100
export SPARK_CPUS=14

# Activate conda environment
eval "$(/scicore/home/meiera/schulz0022/miniforge-pypy3/bin/conda shell.bash hook)"
conda activate gnt

# Change to project directory
cd /scicore/home/meiera/schulz0022/projects/growth-and-temperature

# Set DATA_NOBACKUP environment variable
export DATA_NOBACKUP="/scicore/home/meiera/schulz0022/projects/growth-and-temperature/data_nobackup"

# Run the demeaning using the unified interface
python run.py demean \
    --config orchestration/configs/data.yaml \
    --source modis
