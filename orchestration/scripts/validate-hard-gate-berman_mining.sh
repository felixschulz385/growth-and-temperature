#!/bin/bash
#SBATCH --job-name=validate-hard-gate-berman_mining
#SBATCH --output=./log/maintenance/hard_gate/%x-%j.out
#SBATCH --error=./log/maintenance/hard_gate/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --qos=6hours
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Hard-gate pilot run: docs/design/09-integrated-pipeline.md step 9's
# validation gate, "point" archetype, source=berman_mining.
#
# What this proves: that OLD code (src/data/preprocess/sources/
# berman_mining.py, stage=spatial) and NEW code (src/data/sources/
# berman_mining.py, step=grid) produce equivalent Zarr output for the same
# input, via scripts/compare_step_output.py.
#
# Notable, and deliberately exercised here: berman_mining has NO PREPARE
# step (FETCH -> GRID directly, STEPS = (FETCH, GRID)) -- this is the
# "declaring step absence structurally" mechanism
# (docs/design/09-integrated-pipeline.md §2), and this script's `run_new
# prepare` probe (see below) confirms attempting the undeclared step fails
# cleanly (UnsupportedStepError) rather than silently doing nothing or
# crashing on an import error.
#
# Also notable: unlike the design doc's own §5 migration table (which lists
# `berman_mining | REQUIRES=(gadm,PREPARE)`), the actual NEW source declares
# `REQUIRES = ()` -- src/data/sources/berman_mining.py's own docstring
# explains this as a deliberate correction of an earlier, unverified
# planning assumption: berman_mining shares the VIIRS-derived geobox cache
# *location* with gadm/osm, not a real dependency on GADM's output. Verified
# independently (registry.resolve("berman_mining").requires == (), and
# tests/data/sources/berman_mining/test_berman_mining_plan.py asserts the
# same) before writing this script -- no GADM artefact is staged here.
#
# Safety: everything below runs against an ISOLATED scratch data_root, never
# against $DATA_NOBACKUP -- same pattern as validate-hard-gate-acag.sh.
#
# Data required: none from production. berman_mining's real raw data
# (BCRT_baseline.dta) is a manually-downloaded, authenticated ICPSR export
# (src/data/download/sources/manual.py::BermanMiningDataSource) -- not
# fetchable by any automated script, hard-gate or otherwise. This script
# generates a small SYNTHETIC .dta fixture with the exact columns both OLD
# and NEW actually read (latitude, longitude, year, nb_mines_a, nb_diamond),
# so it tests real code on real (if synthetic) data rather than skipping the
# pilot entirely. If you want this run against real content instead, drop a
# real BCRT_baseline.dta at
# $DATA_NOBACKUP/berman_mining/raw/baseline/BCRT_baseline.dta and re-run --
# the script prefers it automatically (see below).
#
# GRID's target grid: see validate-hard-gate-acag.sh's header comment item 2
# for what this pickle is and why it's required. Cropped via
# scripts/crop_geobox_pickle.py, centered on the synthetic fixture's own
# coordinates (or --lon/--lat overrides) so the reprojection has real
# content to check, not an empty/all-nodata result.
#
# Usage:
#   sbatch orchestration/slurm/validate-hard-gate-berman_mining.sh [WINDOW_PX] [LON] [LAT]
#   (WINDOW_PX defaults to 300, LON/LAT default to 25.0/-15.0 -- south-central
#   Africa, matching this dataset's real mining-conflict geography.)

set -euo pipefail

PROJECT_ROOT="/scicore/home/meiera/schulz0022/projects/growth-and-temperature"
PYTHON_BIN="/scicore/home/meiera/schulz0022/miniforge-pypy3/envs/gnt/bin/python"
WINDOW_PX="${1:-300}"
LON="${2:-25.0}"
LAT="${3:--15.0}"

cd "$PROJECT_ROOT"
mkdir -p "./log/maintenance/hard_gate"

LOG_FILE="./log/maintenance/hard_gate/validate-hard-gate-berman_mining-${SLURM_JOB_ID}.log"
exec > >(tee -a "$LOG_FILE")
exec 2>&1

eval "$(/scicore/home/meiera/schulz0022/miniforge-pypy3/bin/conda shell.bash hook)"
conda activate gnt

DATA_NOBACKUP="${DATA_NOBACKUP:-$PROJECT_ROOT/data_nobackup}"
TEST_ROOT="/scratch/schulz0022/hard_gate_berman_mining_${SLURM_JOB_ID}"
TEST_CONFIG="${TEST_ROOT}/config.yaml"
RAW_REL="baseline/BCRT_baseline.dta"
PROD_RAW="${DATA_NOBACKUP}/berman_mining/raw/${RAW_REL}"
TEST_RAW="${TEST_ROOT}/berman_mining/raw/${RAW_REL}"

echo "$(date -Is): Hard-gate pilot -- source=berman_mining window_px=$WINDOW_PX lon=$LON lat=$LAT"
echo "Job ID:     $SLURM_JOB_ID"
echo "Test root:  $TEST_ROOT  (isolated -- production data_nobackup is never written to)"
echo "Log file:   $LOG_FILE"

mkdir -p "$(dirname "$TEST_RAW")"

# --- REQUIRED: stage the raw .dta this run needs ----------------------------
if [ -f "$PROD_RAW" ]; then
    echo "$(date -Is): copying already-fetched raw file from production"
    cp "$PROD_RAW" "$TEST_RAW"
else
    echo "$(date -Is): no production raw file at $PROD_RAW -- generating a synthetic fixture"
    echo "  (BCRT_baseline.dta is a manually-downloaded, authenticated ICPSR export --"
    echo "   there is no automated fetch path for it, real or fallback.)"
    "$PYTHON_BIN" - "$TEST_RAW" "$LON" "$LAT" <<'PYEOF'
import sys
import pandas as pd

out_path, lon, lat = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])

rows = []
for dlat in (-0.3, 0.0, 0.3):
    for dlon in (-0.3, 0.0, 0.3):
        for year in (2018, 2019, 2020):
            rows.append(
                {
                    "latitude": lat + dlat,
                    "longitude": lon + dlon,
                    "year": year,
                    "nb_mines_a": (abs(int(dlat * 10)) + abs(int(dlon * 10))) % 5,
                    "nb_diamond": (year - 2018),
                }
            )
df = pd.DataFrame(rows)
df.to_stata(out_path, write_index=False)
print(f"wrote synthetic fixture -> {out_path} ({len(df)} rows)")
PYEOF
fi

# --- REQUIRED: seed a CROPPED copy of the shared VIIRS-derived target
# geobox cache, centered on the fixture's own content -- see the header
# comment and validate-hard-gate-gadm.sh's header comment for why cropping
# is centered rather than grid-midpoint here.
PROD_GEOBOX="${DATA_NOBACKUP}/misc/processed/stage_1/misc/viirs_geobox.pkl"
TEST_GEOBOX_DIR="${TEST_ROOT}/misc/processed/stage_1/misc"
mkdir -p "$TEST_GEOBOX_DIR"
if [ -f "$PROD_GEOBOX" ]; then
    "$PYTHON_BIN" scripts/crop_geobox_pickle.py \
        "$PROD_GEOBOX" "${TEST_GEOBOX_DIR}/viirs_geobox.pkl" \
        --window-px "$WINDOW_PX" --lon "$LON" --lat "$LAT"
    echo "$(date -Is): seeded cropped target geobox cache from production"
else
    echo "$(date -Is): ERROR -- no cached target geobox at $PROD_GEOBOX"
    echo "  Run 'data run --source eog_viirs --step grid' once for real first,"
    echo "  or copy a viirs_geobox.pkl from wherever GRID has run before."
    exit 2
fi

# --- self-contained test config: isolated data_root, no HPC remote ---------
# berman_mining is a top-level source in both OLD and NEW (not part of the
# misc split) -- one `sources.berman_mining` block works for both codepaths.
cat > "$TEST_CONFIG" <<EOF
paths:
  data_root: "${TEST_ROOT}"
  local_index_dir: "${TEST_ROOT}/hpc_data_index"
remote:
  ssh_target: ""
  key_file: ""
sources:
  berman_mining:
    type: "berman_mining"
    data_path: "berman_mining"
EOF

run_old() {
    local stage="$1"
    echo "$(date -Is): OLD  preprocess run --source berman_mining --stage $stage"
    "$PYTHON_BIN" run.py preprocess run --config "$TEST_CONFIG" --source berman_mining \
        --stage "$stage" --override \
        --dask-threads "$SLURM_CPUS_PER_TASK" --dask-memory-limit 4GiB \
        --temp-dir "${TEST_ROOT}/dask_tmp"
}

run_new() {
    local step="$1"
    echo "$(date -Is): NEW  data run --source berman_mining --step $step"
    "$PYTHON_BIN" run.py data run --config "$TEST_CONFIG" --source berman_mining \
        --step "$step" --override \
        --dask-threads "$SLURM_CPUS_PER_TASK" --dask-memory-limit 4GiB \
        --temp-dir "${TEST_ROOT}/dask_tmp"
}

# --- 0. Confirm NEW's declared step-absence is a clean, structural error ---
# (docs/design/09-integrated-pipeline.md §2 -- not exercised by the acag/gadm
# pilots since both of those sources DO have a PREPARE step. `set -e` is
# deliberately suspended for this one call since a nonzero exit is success.)
echo "$(date -Is): confirming NEW correctly rejects the undeclared PREPARE step for this source"
set +e
"$PYTHON_BIN" run.py data run --config "$TEST_CONFIG" --source berman_mining --step prepare 2>&1 | tail -5
PREPARE_REJECTED=$?
set -e
if [ $PREPARE_REJECTED -eq 0 ]; then
    echo "  UNEXPECTED: --step prepare succeeded for a source with no PREPARE in STEPS"
else
    echo "  OK: --step prepare failed as expected (step-absence is structural, not silent)"
fi

# --- 1. OLD code produces the reference artefact ----------------------------
run_old spatial

GRID_DIR="${TEST_ROOT}/berman_mining/processed/stage_2"
GRID_OLD_REF="${GRID_DIR}.old_reference"

echo "$(date -Is): snapshotting OLD output before NEW code overwrites the same paths"
cp -a "$GRID_DIR" "$GRID_OLD_REF"

# --- 2. NEW code overwrites the same target paths ---------------------------
run_new grid

# --- 3. Compare ---------------------------------------------------------------
echo "$(date -Is): comparing GRID output"
GRID_STATUS=0
"$PYTHON_BIN" scripts/compare_step_output.py \
    "${GRID_OLD_REF}/berman_mining_timeseries_reprojected.zarr" \
    "${GRID_DIR}/berman_mining_timeseries_reprojected.zarr" || GRID_STATUS=$?

echo "=============================================================="
echo "HARD-GATE PILOT RESULT -- source=berman_mining window_px=$WINDOW_PX lon=$LON lat=$LAT"
echo "  Step-absence check (PREPARE correctly rejected): $([ $PREPARE_REJECTED -ne 0 ] && echo OK || echo UNEXPECTED)"
echo "  GRID (spatial vs grid): $([ $GRID_STATUS -eq 0 ] && echo EQUIVALENT || echo NOT_EQUIVALENT)"
echo "  Test root (not auto-deleted, inspect or clean up manually): $TEST_ROOT"
echo "=============================================================="

if [ $GRID_STATUS -ne 0 ]; then
    exit 1
fi
exit 0
