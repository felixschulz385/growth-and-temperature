#!/bin/bash
#SBATCH --job-name=validate-hard-gate-plad
#SBATCH --output=./log/maintenance/hard_gate/%x-%j.out
#SBATCH --error=./log/maintenance/hard_gate/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --qos=6hours
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Hard-gate pilot run: docs/design/09-integrated-pipeline.md step 9's
# validation gate, "point" archetype, source=plad.
#
# What this proves: that OLD code (src/data/preprocess/sources/plad.py,
# stage=spatial) and NEW code (src/data/sources/plad.py, step=grid) produce
# equivalent Zarr output for the same input, via
# scripts/compare_step_output.py. Also exercises NEW's `REQUIRES =
# (("gadm", PipelineStep.PREPARE),)` -- confirmed real (not a stale planning
# assumption the way berman_mining's turned out to be, see
# validate-hard-gate-berman_mining.sh) and enforced by
# src/cli/data/handlers.py::_check_requires BEFORE the source is even
# constructed, for every subcommand including `plan`/`index`, not just
# `run`. PLAD depends on GADM's PREPARE output specifically (the simplified
# ADM1/ADM2 .gpkg vector files), not GADM's GRID/rasterized output --
# confirmed by reading _resolve_gadm_files_from_preprocessed in both OLD and
# NEW, which read misc/processed/stage_1/gadm/gadm_levelADM_{1,2}_simplified.gpkg.
#
# PLAD has no PREPARE step of its own (STEPS = (FETCH, GRID) in NEW; OLD's
# only stage is "spatial") -- this script does not attempt one.
#
# Two config/path quirks worth knowing, both preserved as-is (not "fixed"
# here, since fixing them is out of this pilot's scope):
#   - GRID's output path hardcodes a literal "plad" prefix in BOTH OLD and
#     NEW, ignoring whatever `data_path` is actually configured
#     (confirmed identical in both -- not a discrepancy, just a shared quirk).
#   - The completion-index parquet path is HARDCODED in OLD
#     (<hpc_root>/hpc_data_index/parquet_plad.parquet, ignoring any
#     configured local_index_dir) but config-driven in NEW
#     (<local_index_dir>/parquet_plad.parquet). This script's TEST_CONFIG
#     points local_index_dir at <data_root>/hpc_data_index specifically so
#     both resolve to the same file -- a different local_index_dir would
#     make OLD and NEW read two different (and for OLD, un-seeded) indexes.
#
# Safety: everything below runs against an ISOLATED scratch data_root, never
# against $DATA_NOBACKUP -- same pattern as validate-hard-gate-acag.sh.
#
# Data required: none from production. PLAD's real raw data
# (PLAD_April_2024.dta -- a tab-delimited text file despite the .dta
# extension, from Harvard Dataverse) is public but this script has no
# network access guarantee either way, so it always generates a small
# SYNTHETIC fixture with the 5 columns both OLD and NEW actually read
# (gid_0, gid_1, gid_2, startyear, endyear), plus matching tiny synthetic
# GADM ADM1/ADM2 .gpkg files (bypassing a real GADM PREPARE run -- this
# pilot is about PLAD's own code, not re-validating GADM's, which
# validate-hard-gate-gadm.sh already covers).
#
# GRID's target grid: see validate-hard-gate-acag.sh's header comment item 2.
# Cropped via scripts/crop_geobox_pickle.py, centered on the synthetic
# fixture's own coordinates.
#
# Usage:
#   sbatch orchestration/slurm/validate-hard-gate-plad.sh [ADMIN_LEVEL] [WINDOW_PX] [LON] [LAT]
#   (ADMIN_LEVEL defaults to 1, WINDOW_PX to 300, LON/LAT to 10.0/50.0 --
#   central Europe, same default as validate-hard-gate-gadm.sh.)

set -euo pipefail

PROJECT_ROOT="/scicore/home/meiera/schulz0022/projects/growth-and-temperature"
PYTHON_BIN="/scicore/home/meiera/schulz0022/miniforge-pypy3/envs/gnt/bin/python"
ADMIN_LEVEL="${1:-1}"
WINDOW_PX="${2:-300}"
LON="${3:-10.0}"
LAT="${4:-50.0}"

cd "$PROJECT_ROOT"
mkdir -p "./log/maintenance/hard_gate"

LOG_FILE="./log/maintenance/hard_gate/validate-hard-gate-plad-${SLURM_JOB_ID}.log"
exec > >(tee -a "$LOG_FILE")
exec 2>&1

eval "$(/scicore/home/meiera/schulz0022/miniforge-pypy3/bin/conda shell.bash hook)"
conda activate gnt

DATA_NOBACKUP="${DATA_NOBACKUP:-$PROJECT_ROOT/data_nobackup}"
TEST_ROOT="/scratch/schulz0022/hard_gate_plad_${SLURM_JOB_ID}"
TEST_CONFIG="${TEST_ROOT}/config.yaml"

echo "$(date -Is): Hard-gate pilot -- source=plad admin_level=$ADMIN_LEVEL window_px=$WINDOW_PX lon=$LON lat=$LAT"
echo "Job ID:     $SLURM_JOB_ID"
echo "Test root:  $TEST_ROOT  (isolated -- production data_nobackup is never written to)"
echo "Log file:   $LOG_FILE"

mkdir -p "${TEST_ROOT}/plad/raw" "${TEST_ROOT}/hpc_data_index" "${TEST_ROOT}/misc/processed/stage_1/gadm"

# --- REQUIRED: synthetic PLAD raw fixture + matching GADM ADM1/ADM2 gpkgs --
echo "$(date -Is): generating synthetic PLAD fixture + matching GADM PREPARE stand-in"
"$PYTHON_BIN" - "$TEST_ROOT" "$LON" "$LAT" "$ADMIN_LEVEL" <<'PYEOF'
import os
import sys

import geopandas as gpd
import pandas as pd
from shapely.geometry import MultiPolygon, box

test_root, lon, lat, admin_level = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])

gid_0, gid_1, gid_2 = "DEU", "DEU.1_1", "DEU.1.1_1"

# Raw PLAD fixture: tab-delimited despite the .dta extension (see
# src/data/preprocess/sources/plad.py's own comment on this).
plad_df = pd.DataFrame(
    [
        {"gid_0": gid_0, "gid_1": gid_1, "gid_2": gid_2, "startyear": 2018, "endyear": 2020},
        {"gid_0": gid_0, "gid_1": gid_1, "gid_2": gid_2, "startyear": 2021, "endyear": 2022},
    ]
)
plad_path = os.path.join(test_root, "plad", "raw", "PLAD_April_2024.dta")
plad_df.to_csv(plad_path, sep="\t", index=False)
print(f"wrote PLAD fixture -> {plad_path}")

# Matching GADM PREPARE stand-in (bypasses running gadm's own PREPARE --
# validate-hard-gate-gadm.sh already validates that step in isolation).
gadm_dir = os.path.join(test_root, "misc", "processed", "stage_1", "gadm")
os.makedirs(gadm_dir, exist_ok=True)

# MultiPolygon, not Polygon -- real GADM data is always MultiPolygon (even
# single-part regions), and OLD's _rasterize_panel iterates `mgeom.geoms`
# unconditionally; a plain Polygon fixture here raises
# AttributeError: 'Polygon' object has no attribute 'geoms'" (found by
# actually running this fixture against the real code before handing this
# script off, not assumed).
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

# --- REQUIRED: parquet index -- see header comment on the hardcoded-vs-
# config-driven path quirk this script's local_index_dir choice resolves ---
"$PYTHON_BIN" - "${TEST_ROOT}/hpc_data_index/parquet_plad.parquet" <<'PYEOF'
import sys
import pandas as pd

out_path = sys.argv[1]
pd.DataFrame([{"relative_path": "PLAD_April_2024.dta", "status_category": "completed"}]).to_parquet(out_path)
print(f"wrote index -> {out_path}")
PYEOF

# --- REQUIRED: seed a CROPPED copy of the shared VIIRS-derived target
# geobox cache, centered on the fixture's own content --------------------
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
# `gadm:` is only here to satisfy NEW's REQUIRES pre-flight check
# (_check_requires) -- data_path/namespace must be set explicitly, since
# that check builds a bare SourceConfig without running GadmSource's own
# defaulting (see header comment).
cat > "$TEST_CONFIG" <<EOF
paths:
  data_root: "${TEST_ROOT}"
  local_index_dir: "${TEST_ROOT}/hpc_data_index"
remote:
  ssh_target: ""
  key_file: ""
sources:
  plad:
    type: "plad"
    data_path: "plad"
    admin_level: ${ADMIN_LEVEL}
    year_range: [2018, 2022]
  gadm:
    type: "gadm"
    data_path: "misc"
    namespace: "gadm"
EOF

run_old() {
    echo "$(date -Is): OLD  preprocess run --source plad --stage spatial --admin-level $ADMIN_LEVEL"
    "$PYTHON_BIN" run.py preprocess run --config "$TEST_CONFIG" --source plad \
        --stage spatial --admin-level "$ADMIN_LEVEL" --override \
        --dask-threads "$SLURM_CPUS_PER_TASK" --dask-memory-limit 4GiB \
        --temp-dir "${TEST_ROOT}/dask_tmp"
}

run_new() {
    echo "$(date -Is): NEW  data run --source plad --step grid"
    "$PYTHON_BIN" run.py data run --config "$TEST_CONFIG" --source plad \
        --step grid --override \
        --dask-threads "$SLURM_CPUS_PER_TASK" --dask-memory-limit 4GiB \
        --temp-dir "${TEST_ROOT}/dask_tmp"
}

# --- 1. OLD code produces the reference artefact ----------------------------
run_old

GRID_DIR="${TEST_ROOT}/plad/processed/stage_2"
GRID_OLD_REF="${GRID_DIR}.old_reference"

echo "$(date -Is): snapshotting OLD output before NEW code overwrites the same paths"
cp -a "$GRID_DIR" "$GRID_OLD_REF"

# --- 2. NEW code overwrites the same target paths ---------------------------
run_new

# --- 3. Compare ---------------------------------------------------------------
echo "$(date -Is): comparing GRID output"
GRID_STATUS=0
"$PYTHON_BIN" scripts/compare_step_output.py \
    "${GRID_OLD_REF}/plad_adm${ADMIN_LEVEL}_timeseries_reprojected.zarr" \
    "${GRID_DIR}/plad_adm${ADMIN_LEVEL}_timeseries_reprojected.zarr" || GRID_STATUS=$?

echo "=============================================================="
echo "HARD-GATE PILOT RESULT -- source=plad admin_level=$ADMIN_LEVEL window_px=$WINDOW_PX lon=$LON lat=$LAT"
echo "  GRID (spatial vs grid): $([ $GRID_STATUS -eq 0 ] && echo EQUIVALENT || echo NOT_EQUIVALENT)"
echo "  Test root (not auto-deleted, inspect or clean up manually): $TEST_ROOT"
echo "=============================================================="

if [ $GRID_STATUS -ne 0 ]; then
    exit 1
fi
exit 0
