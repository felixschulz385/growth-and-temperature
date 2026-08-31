#!/bin/bash
#SBATCH --job-name=validate-hard-gate-snl_mining
#SBATCH --output=./log/maintenance/hard_gate/%x-%j.out
#SBATCH --error=./log/maintenance/hard_gate/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --qos=6hours
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Hard-gate pilot run: docs/design/09-integrated-pipeline.md step 9's
# validation gate, "point" archetype, source=snl_mining -- the third point
# source (berman_mining: FETCH+GRID, plad: FETCH+GRID; snl_mining: PREPARE+
# GRID, no FETCH), each declaring a genuinely different step-absence
# pattern worth its own pilot (docs/design/09-integrated-pipeline.md §2,
# §6).
#
# snl_mining has NO FETCH step (STEPS = (PREPARE, GRID)) -- its raw data is
# a manual S&P Global .xls export + an OpenAI-enrichment notebook, declared
# absent rather than invented (see src/data/sources/snl_mining/notebooks/).
# This script bypasses the notebook entirely with a small SYNTHETIC
# stage-0 DuckDB matching the exact minimal schema both OLD and NEW read
# (properties: property_id, latitude, longitude, actual_start_up_year,
# actual_closure_year -- property_llm_years is optional and auto-detected
# absent by both codepaths, confirmed by reading both).
#
# PREPARE (both OLD's single "spatial" stage, which does prepare-then-grid
# internally in one call, and NEW's separate `--step prepare`) writes
# &lt;data_root&gt;/snl_mining/processed/stage_1/snl_mining_prepared.duckdb --
# NOT comparable via scripts/compare_step_output.py's zarr/gpkg/parquet
# branches, so this migration's compare tool was extended with DuckDB
# support (compare_duckdb) specifically to make this pilot possible.
#
# REQUIRES = (("gadm", PipelineStep.PREPARE),) -- same as plad (GADM's
# simplified ADM1/ADM2 .gpkg vector output, not its rasterized grid),
# enforced the same way (see validate-hard-gate-plad.sh's header comment on
# _check_requires and the explicit data_path/namespace requirement on the
# `gadm:` config block).
#
# IMPORTANT, found by actually running this locally before handing it off:
# with no explicit `year_range`, both OLD and NEW infer year bounds from
# MIN(actual_start_up_year)..MAX(actual_closure_year) across the ENTIRE
# properties table -- on real production data this span is pathological
# (1554-2026, uncleaned source-data tails), which would make both PREPARE's
# per-year expansion and GRID's per-year tile loop extremely slow for
# reasons unrelated to code correctness. This script always sets an
# explicit, narrow `year_range` for exactly this reason (mirroring why the
# acag/gadm scripts crop the geobox).
#
# Also note: cropping the geobox (below) only bounds GRID's cost, not
# PREPARE's -- PREPARE operates on the whole properties/GADM tables
# regardless of the target geobox's extent; only `year_range` and the
# synthetic fixture's own small row count bound PREPARE's cost here.
#
# Safety: everything below runs against an ISOLATED scratch data_root, never
# against $DATA_NOBACKUP -- same pattern as validate-hard-gate-acag.sh.
#
# Prerequisite: DuckDB's `spatial` extension must already be cached locally
# (run ./orchestration/slurm/bootstrap_duckdb_extensions.sh once from a login
# node first -- see environment.yml's Install: instructions). Compute nodes
# have no internet access, so `INSTALL spatial` fails here otherwise
# (`IOException: ... Connection timed out` fetching extensions.duckdb.org --
# found by actually running this pilot on SLURM, not assumed).
#
# Usage:
#   sbatch orchestration/slurm/validate-hard-gate-snl_mining.sh [WINDOW_PX] [LON] [LAT]
#   (WINDOW_PX defaults to 300, LON/LAT default to 10.0/50.0 -- central
#   Europe, same default as validate-hard-gate-gadm.sh/-plad.sh.)

set -euo pipefail

PROJECT_ROOT="/scicore/home/meiera/schulz0022/projects/growth-and-temperature"
PYTHON_BIN="/scicore/home/meiera/schulz0022/miniforge-pypy3/envs/gnt/bin/python"
WINDOW_PX="${1:-300}"
LON="${2:-10.0}"
LAT="${3:-50.0}"

cd "$PROJECT_ROOT"
mkdir -p "./log/maintenance/hard_gate"

LOG_FILE="./log/maintenance/hard_gate/validate-hard-gate-snl_mining-${SLURM_JOB_ID}.log"
exec > >(tee -a "$LOG_FILE")
exec 2>&1

eval "$(/scicore/home/meiera/schulz0022/miniforge-pypy3/bin/conda shell.bash hook)"
conda activate gnt

DATA_NOBACKUP="${DATA_NOBACKUP:-$PROJECT_ROOT/data_nobackup}"
TEST_ROOT="/scratch/schulz0022/hard_gate_snl_mining_${SLURM_JOB_ID}"
TEST_CONFIG="${TEST_ROOT}/config.yaml"

echo "$(date -Is): Hard-gate pilot -- source=snl_mining window_px=$WINDOW_PX lon=$LON lat=$LAT"
echo "Job ID:     $SLURM_JOB_ID"
echo "Test root:  $TEST_ROOT  (isolated -- production data_nobackup is never written to)"
echo "Log file:   $LOG_FILE"

mkdir -p "${TEST_ROOT}/snl_mining/processed/stage_0/manual_xls" \
         "${TEST_ROOT}/misc/processed/stage_1/gadm" \
         "${TEST_ROOT}/misc/processed/stage_1/misc"

# --- REQUIRED: synthetic stage-0 DuckDB (bypasses the manual notebook) +
# matching GADM ADM1/ADM2 PREPARE stand-in --------------------------------
echo "$(date -Is): generating synthetic stage-0 DuckDB + matching GADM PREPARE stand-in"
"$PYTHON_BIN" - "$TEST_ROOT" "$LON" "$LAT" <<'PYEOF'
import os
import sys

import duckdb
import geopandas as gpd
from shapely.geometry import MultiPolygon, box

test_root, lon, lat = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])

duckdb_path = os.path.join(test_root, "snl_mining", "processed", "stage_0", "manual_xls", "snl_mining_manual_export.duckdb")
con = duckdb.connect(duckdb_path)
try:
    con.execute("LOAD spatial;")
except Exception:
    try:
        con.execute("INSTALL spatial;")
        con.execute("LOAD spatial;")
    except Exception as exc:
        raise RuntimeError(
            "DuckDB 'spatial' extension is not cached locally and this node has no "
            "internet access to download it. Run "
            "./orchestration/slurm/bootstrap_duckdb_extensions.sh once from a login "
            "node first (see environment.yml's Install: instructions)."
        ) from exc
con.execute(
    "CREATE TABLE properties (property_id VARCHAR, latitude DOUBLE, longitude DOUBLE, "
    "actual_start_up_year DOUBLE, actual_closure_year DOUBLE)"
)
con.execute(
    "INSERT INTO properties VALUES (?, ?, ?, ?, ?), (?, ?, ?, ?, ?)",
    ["m1", lat, lon, 2018, 2020, "m2", lat + 0.05, lon + 0.05, 2019, None],
)
con.close()
print(f"wrote synthetic stage-0 DuckDB -> {duckdb_path}")

gid_0, gid_1, gid_2 = "DEU", "DEU.1_1", "DEU.1.1_1"
gadm_dir = os.path.join(test_root, "misc", "processed", "stage_1", "gadm")
adm1 = gpd.GeoDataFrame(
    [{"GID_0": gid_0, "GID_1": gid_1, "geometry": MultiPolygon([box(lon - 0.5, lat - 0.5, lon + 0.5, lat + 0.5)])}],
    crs="EPSG:4326",
)
adm1.to_file(os.path.join(gadm_dir, "gadm_levelADM_1_simplified.gpkg"), driver="GPKG")
adm2 = gpd.GeoDataFrame(
    [{"GID_0": gid_0, "GID_1": gid_1, "GID_2": gid_2, "geometry": MultiPolygon([box(lon - 0.5, lat - 0.5, lon + 0.5, lat + 0.5)])}],
    crs="EPSG:4326",
)
adm2.to_file(os.path.join(gadm_dir, "gadm_levelADM_2_simplified.gpkg"), driver="GPKG")
print(f"wrote GADM ADM1/ADM2 fixtures -> {gadm_dir}")
PYEOF

# --- REQUIRED: seed a CROPPED copy of the shared VIIRS-derived target
# geobox cache, centered on the fixture's own content (see header comment).
PROD_GEOBOX="${DATA_NOBACKUP}/misc/processed/stage_1/misc/viirs_geobox.pkl"
if [ -f "$PROD_GEOBOX" ]; then
    "$PYTHON_BIN" scripts/crop_geobox_pickle.py \
        "$PROD_GEOBOX" "${TEST_ROOT}/misc/processed/stage_1/misc/viirs_geobox.pkl" \
        --window-px "$WINDOW_PX" --lon "$LON" --lat "$LAT"
    echo "$(date -Is): seeded cropped target geobox cache from production"
else
    echo "$(date -Is): ERROR -- no cached target geobox at $PROD_GEOBOX"
    echo "  Run 'data run --source eog_viirs --step grid' once for real first,"
    echo "  or copy a viirs_geobox.pkl from wherever GRID has run before."
    exit 2
fi

# --- self-contained test config: isolated data_root, no HPC remote ---------
cat > "$TEST_CONFIG" <<EOF
paths:
  data_root: "${TEST_ROOT}"
  local_index_dir: "${TEST_ROOT}/hpc_data_index"
remote:
  ssh_target: ""
  key_file: ""
sources:
  snl_mining:
    type: "snl_mining"
    data_path: "snl_mining"
    year_range: [2018, 2020]
  gadm:
    type: "gadm"
    data_path: "misc"
    namespace: "gadm"
EOF

run_old() {
    echo "$(date -Is): OLD  preprocess run --source snl_mining --stage spatial"
    "$PYTHON_BIN" run.py preprocess run --config "$TEST_CONFIG" --source snl_mining \
        --stage spatial --override \
        --dask-threads "$SLURM_CPUS_PER_TASK" --dask-memory-limit 4GiB \
        --temp-dir "${TEST_ROOT}/dask_tmp"
}

run_new() {
    local step="$1"
    echo "$(date -Is): NEW  data run --source snl_mining --step $step"
    "$PYTHON_BIN" run.py data run --config "$TEST_CONFIG" --source snl_mining \
        --step "$step" --override \
        --dask-threads "$SLURM_CPUS_PER_TASK" --dask-memory-limit 4GiB \
        --temp-dir "${TEST_ROOT}/dask_tmp"
}

# --- 1. OLD code produces the reference artefacts (one call does both
# prepare-equivalent and grid-equivalent work -- OLD has no separate
# "prepare" stage for this source) -------------------------------------------
run_old

PREPARED_DB="${TEST_ROOT}/snl_mining/processed/stage_1/snl_mining_prepared.duckdb"
GRID_DIR="${TEST_ROOT}/snl_mining/processed/stage_2"
PREPARED_DB_OLD_REF="${TEST_ROOT}/snl_mining/processed/stage_1/snl_mining_prepared.old_reference.duckdb"
GRID_OLD_REF="${GRID_DIR}.old_reference"

echo "$(date -Is): snapshotting OLD output before NEW code overwrites the same paths"
cp "$PREPARED_DB" "$PREPARED_DB_OLD_REF"
cp -a "$GRID_DIR" "$GRID_OLD_REF"

# --- 2. NEW code overwrites the same target paths ---------------------------
run_new prepare
run_new grid

# --- 3. Compare ---------------------------------------------------------------
echo "$(date -Is): comparing PREPARE output (DuckDB)"
PREPARE_STATUS=0
"$PYTHON_BIN" scripts/compare_step_output.py \
    "$PREPARED_DB_OLD_REF" "$PREPARED_DB" || PREPARE_STATUS=$?

echo "$(date -Is): comparing GRID output"
GRID_STATUS=0
"$PYTHON_BIN" scripts/compare_step_output.py \
    "${GRID_OLD_REF}/snl_mining_timeseries_reprojected.zarr" \
    "${GRID_DIR}/snl_mining_timeseries_reprojected.zarr" || GRID_STATUS=$?

echo "=============================================================="
echo "HARD-GATE PILOT RESULT -- source=snl_mining window_px=$WINDOW_PX lon=$LON lat=$LAT"
echo "  PREPARE (spatial-internal vs prepare): $([ $PREPARE_STATUS -eq 0 ] && echo EQUIVALENT || echo NOT_EQUIVALENT)"
echo "  GRID    (spatial vs grid):             $([ $GRID_STATUS -eq 0 ] && echo EQUIVALENT || echo NOT_EQUIVALENT)"
echo "  Test root (not auto-deleted, inspect or clean up manually): $TEST_ROOT"
echo "=============================================================="

if [ $PREPARE_STATUS -ne 0 ] || [ $GRID_STATUS -ne 0 ]; then
    exit 1
fi
exit 0
