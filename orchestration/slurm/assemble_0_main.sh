#!/bin/bash
#SBATCH --job-name=assemble_0_main
#SBATCH --output=./log/slurm-%j.out
#SBATCH --error=./log/slurm-%j.err
#SBATCH --partition=scicore
#SBATCH --time=1-00:00:00
#SBATCH --qos=1day
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G

# Activate conda environment
eval "$(/scicore/home/meiera/schulz0022/miniforge-pypy3/bin/conda shell.bash hook)"
conda activate gnt

# Load necessary modules
module load Java/21.0.2

# Export SLURM variables for Spark configuration
export SPARK_CPUS=${SLURM_CPUS_PER_TASK:-32}
export SPARK_MEMORY_GB=$((${SLURM_MEM_PER_NODE:-262144} / 1024))

# Run the assembly script
/scicore/home/meiera/schulz0022/miniforge-pypy3/envs/gnt/bin/python /scicore/home/meiera/schulz0022/projects/growth-and-temperature/gnt/data/assemble/0_main.py