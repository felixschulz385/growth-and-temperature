#!/bin/bash
#SBATCH --job-name=assemble_modis
#SBATCH --output=./log/slurm-%j.log
#SBATCH --error=./log/slurm-%j.err
#SBATCH --partition=scicore
# SBATCH --time=1-00:00:00
# SBATCH --qos=1day
#SBATCH --time=00:30:00
#SBATCH --qos=30min
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Activate conda environment
eval "$(/scicore/home/meiera/schulz0022/miniforge-pypy3/bin/conda shell.bash hook)"
conda activate gnt

# Change to project directory
cd /scicore/home/meiera/schulz0022/projects/growth-and-temperature

# Set DATA_NOBACKUP environment variable
export DATA_NOBACKUP="/scicore/home/meiera/schulz0022/projects/growth-and-temperature/data_nobackup"

# Run the assembly using the unified interface
python run.py assemble \
    --config orchestration/configs/data.yaml \
    --source modis