#!/usr/bin/env bash
# Pre-install DuckDB extensions this repo needs into the user's local
# extension cache (~/.duckdb/extensions by default), so later `LOAD spatial`
# calls on SLURM compute nodes -- which have no internet access -- find the
# extension already on disk instead of trying to fetch it from
# extensions.duckdb.org and failing/timing out.
#
# Run once, from a node WITH internet access (e.g. the login node), any time
# after creating/updating the `src` conda environment. $HOME is shared
# between login and compute nodes on scicore, so one run here covers every
# later SLURM job -- see snl_mining's DuckDB usage
# (src/data/sources/snl_mining/source.py, src/data/preprocess/sources/
# snl_mining.py, scripts/compare_step_output.py's compare_duckdb) and
# orchestration/slurm/validate-hard-gate-snl_mining.sh, which is what
# surfaced this gap when run for real (`IOException: ... Connection timed
# out` trying to download the spatial extension from a compute node).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$PROJECT_ROOT"

eval "$(/scicore/home/meiera/schulz0022/miniforge-pypy3/bin/conda shell.bash hook)"
conda activate src

python -c "
import duckdb

con = duckdb.connect()
con.execute('INSTALL spatial;')
con.execute('LOAD spatial;')
print('duckdb spatial extension installed and loaded successfully')
"
echo "duckdb spatial extension cached in ~/.duckdb/extensions -- SLURM compute-node jobs can now LOAD it offline"
