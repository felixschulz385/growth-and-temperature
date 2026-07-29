#!/bin/bash
#SBATCH --job-name=modis-preprocess-spatial
#SBATCH --output=./log/preprocess/modis/%x-%j.out
#SBATCH --error=./log/preprocess/modis/%x-%j.err
#SBATCH --time=1-00:00:00
#SBATCH --qos=1day
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G

# Mirrors glass-modis-preprocess-spatial.sh / acag-preprocess-spatial.sh
# exactly (conda-env activation, memory-limit calculation, `preprocess run`
# invocation pattern). Unlike stage "annual" (orchestration/scripts/modis-
# ingest-annual.sh), this is an ordinary compute-node SLURM job -- no
# network access needed, it reprojects the annual composites that stage
# "annual" already pushed to scicore (docs/design/08-hpc-transfer.md).
#
# Its only new precondition versus every other source's spatial stage is
# operational, not code: the expected local annual zarrs must already be
# present, i.e. the transfer completed
# (`run.py preprocess transfer --source modis --stage annual` exited 0).

mkdir -p "./log/preprocess/modis"

# Activate conda environment
eval "$(/scicore/home/meiera/schulz0022/miniforge-pypy3/bin/conda shell.bash hook)"
conda activate src

# Calculate memory limit (leave some buffer for system - 60% of allocated)
MEMORY_LIMIT_GB=$(echo "scale=0; $SLURM_MEM_PER_NODE * 0.6 / 1024" | bc)

/scicore/home/meiera/schulz0022/miniforge-pypy3/envs/src/bin/python "/scicore/home/meiera/schulz0022/projects/growth-and-temperature/run.py" preprocess run \
    --config "/scicore/home/meiera/schulz0022/projects/growth-and-temperature/orchestration/configs/data.yaml" \
    --source modis \
    --stage spatial \
    --dask-threads $SLURM_CPUS_PER_TASK \
    --dask-memory-limit "${MEMORY_LIMIT_GB}GiB" \
    --temp-dir "/scratch/schulz0022/modis_${SLURM_JOB_ID}" \
    --dashboard-port 8787 \
    --debug
