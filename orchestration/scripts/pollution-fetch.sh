#!/bin/bash
# Hand-maintained, like modis-fetch.sh. FETCH for the pollution-institutions
# follow-up sources (docs/successors/pollution-data-options.md):
#   no2      Anenberg et al. surface NO2, figshare, ~8 GB
#   ghs_pop  GHS-POP R2023A 1 km epochs 1990-2025, JRC FTP, ~2.4 GB
# Too large for the compute nodes' proxy (CLAUDE.md § Environment), so run it
# on the transfer node -- NOT via sbatch:
#   ssh transfer12.scicore.unibas.ch
#   bash /scicore/home/meiera/schulz0022/projects/growth-and-temperature/orchestration/scripts/pollution-fetch.sh
# The transfer node mounts the same /scicore/home, so FETCH writes straight
# into data_nobackup/raw/ and no `data transfer` push is needed. Re-running
# only fetches what is missing. Then submit PREPARE from a login node:
#   data run --source no2 --step prepare --slurm
#   data run --source ghs_pop --step prepare --slurm

set -euo pipefail

cd /scicore/home/meiera/schulz0022/projects/growth-and-temperature
mkdir -p "./log/preprocess/pollution"

PY=/scicore/home/meiera/schulz0022/miniforge-pypy3/envs/gnt/bin/python
CFG=/scicore/home/meiera/schulz0022/projects/growth-and-temperature/orchestration/configs/data.yaml

for source in no2 ghs_pop; do
    echo "== fetch $source"
    "$PY" -m src.cli data run --config "$CFG" --source "$source" --step fetch "$@"
done
echo "fetch complete"
