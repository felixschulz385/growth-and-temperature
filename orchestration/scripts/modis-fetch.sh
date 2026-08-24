#!/bin/bash
# Hand-maintained (not generated -- MODIS's is the only FETCH step in this
# repo that needs its own script, see orchestration/configs/slurm_jobs.yaml's
# header comment).
#
# modis-fetch: runs on whatever host has outbound internet egress (a scicore
# login/transfer node, a workstation, a cloud VM) -- NOT submitted via sbatch.
# After it completes, submit MODIS's PREPARE (SLURM) via:
#   data run --source modis --step prepare --slurm
# or, to also pull in a full dependency chain:
#   data run --source modis --step prepare --slurm --chain
# See docs/design/09-integrated-pipeline.md §9 / docs/design/08-hpc-transfer.md.

set -euo pipefail

mkdir -p "./log/preprocess/modis"
cd /scicore/home/meiera/schulz0022/projects/growth-and-temperature

# --override toggle -- either of these works:
#   sbatch <this script>.sh --override
#   sbatch --export=ALL,PIPELINE_OVERRIDE=1 <this script>.sh
OVERRIDE_FLAG=""
for _arg in "$@"; do
    if [ "$_arg" = "--override" ]; then
        OVERRIDE_FLAG="--override"
    fi
done
if [ -n "${PIPELINE_OVERRIDE:-}" ]; then
    OVERRIDE_FLAG="--override"
fi

/scicore/home/meiera/schulz0022/miniforge-pypy3/envs/src/bin/python "/scicore/home/meiera/schulz0022/projects/growth-and-temperature/run.py" data run \
    --config "/scicore/home/meiera/schulz0022/projects/growth-and-temperature/orchestration/configs/data.yaml" \
    --source modis \
    --step fetch \
    $OVERRIDE_FLAG \
    --debug

echo "fetch complete -- pushing results to scicore"
/scicore/home/meiera/schulz0022/miniforge-pypy3/envs/src/bin/python "/scicore/home/meiera/schulz0022/projects/growth-and-temperature/run.py" data transfer \
    --config "/scicore/home/meiera/schulz0022/projects/growth-and-temperature/orchestration/configs/data.yaml" \
    --source modis \
    --step fetch \
    --direction push
