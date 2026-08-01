#!/bin/bash
#SBATCH --job-name=validate-hard-gate-osm
#SBATCH --output=./log/maintenance/hard_gate/%x-%j.out
#SBATCH --error=./log/maintenance/hard_gate/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --qos=6hours
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Hard-gate pilot run: docs/design/09-integrated-pipeline.md step 9's
# validation gate, "vector" archetype, source=osm -- the second vector
# source (gadm is validate-hard-gate-gadm.sh); OSM's real content (land
# polygons, one shapefile) is spatially and structurally different enough
# from GADM's (.gpkg, multiple admin levels) to be worth its own pilot
# rather than assuming the archetype is fully covered by gadm alone.
#
# What this proves: that OLD code (src/data/preprocess/sources/misc.py,
# subsource=osm, stage=vector/spatial) and NEW code (src/data/sources/
# misc/osm.py, step=prepare/grid) produce equivalent GeoPackage (PREPARE)
# and Zarr (GRID) output for the same input, via
# scripts/compare_step_output.py.
#
# Behavioral quirk deliberately NOT "fixed" here, exercised as-is: OLD's
# GRID rasterization (_rasterize_osm_target) has no existence-check guard at
# all -- it always re-rasterizes regardless of --override. NEW added a
# proper is_complete() skip guard. Both this script's runs always pass
# --override, so the difference is invisible here; it would only matter for
# a non-override production re-run, out of this pilot's scope.
#
# Safety: everything below runs against an ISOLATED scratch data_root, never
# against $DATA_NOBACKUP -- same pattern as validate-hard-gate-acag.sh.
#
# Data required:
#   1. The one already-fetched raw OSM land-polygons zip under
#      $DATA_NOBACKUP/misc/raw/osm/land-polygons-complete-4326.zip (note
#      "raw/" -- both OLD's _resolve_source_file_path and NEW's
#      layout.raw_root() insert it, confirmed by reading both, same as
#      GADM). The real archive is 1GB+; this script does NOT attempt a
#      network fallback fetch (osmdata.openstreetmap.de serves the full
#      complete package only, no small-subset option) -- if the production
#      file isn't there, this script builds a small SYNTHETIC shapefile-in-
#      zip fixture instead, so the pilot still exercises real code on real
#      (if synthetic) data rather than failing outright.
#   2. GRID's target grid -- see validate-hard-gate-acag.sh's header comment
#      item 2. Cropped via scripts/crop_geobox_pickle.py, centered on real
#      land (--lon/--lat) since OSM land polygons are spatially sparse like
#      GADM, not globally continuous like ACAG's PM2.5.
#
# Usage:
#   sbatch orchestration/slurm/validate-hard-gate-osm.sh [WINDOW_PX] [LON] [LAT]
#   (WINDOW_PX defaults to 300, LON/LAT default to 10.0/50.0 -- central
#   Europe, same default as validate-hard-gate-gadm.sh.)

set -euo pipefail

PROJECT_ROOT="/scicore/home/meiera/schulz0022/projects/growth-and-temperature"
PYTHON_BIN="/scicore/home/meiera/schulz0022/miniforge-pypy3/envs/src/bin/python"
WINDOW_PX="${1:-300}"
LON="${2:-10.0}"
LAT="${3:-50.0}"

cd "$PROJECT_ROOT"
mkdir -p "./log/maintenance/hard_gate"

LOG_FILE="./log/maintenance/hard_gate/validate-hard-gate-osm-${SLURM_JOB_ID}.log"
exec > >(tee -a "$LOG_FILE")
exec 2>&1

eval "$(/scicore/home/meiera/schulz0022/miniforge-pypy3/bin/conda shell.bash hook)"
conda activate src

DATA_NOBACKUP="${DATA_NOBACKUP:-$PROJECT_ROOT/data_nobackup}"
TEST_ROOT="/scratch/schulz0022/hard_gate_osm_${SLURM_JOB_ID}"
TEST_CONFIG="${TEST_ROOT}/config.yaml"
OSM_URL="https://osmdata.openstreetmap.de/download/land-polygons-complete-4326.zip"
RAW_REL="osm/land-polygons-complete-4326.zip"
PROD_RAW="${DATA_NOBACKUP}/misc/raw/${RAW_REL}"
TEST_RAW="${TEST_ROOT}/misc/raw/${RAW_REL}"

echo "$(date -Is): Hard-gate pilot -- source=osm window_px=$WINDOW_PX lon=$LON lat=$LAT"
echo "Job ID:     $SLURM_JOB_ID"
echo "Test root:  $TEST_ROOT  (isolated -- production data_nobackup is never written to)"
echo "Log file:   $LOG_FILE"

mkdir -p "$(dirname "$TEST_RAW")" "${TEST_ROOT}/hpc_data_index"

# --- REQUIRED: stage the raw OSM zip this run needs -------------------------
if [ -f "$PROD_RAW" ]; then
    echo "$(date -Is): copying already-fetched raw file from production"
    cp "$PROD_RAW" "$TEST_RAW"
else
    echo "$(date -Is): no production raw file at $PROD_RAW -- generating a synthetic fixture"
    echo "  (the real archive is 1GB+ with no small-subset download option;"
    echo "   this builds a small shapefile-in-zip fixture with the same internal shape instead.)"
    "$PYTHON_BIN" - "$TEST_RAW" "$LON" "$LAT" <<'PYEOF'
import os
import shutil
import sys
import tempfile
import zipfile

import geopandas as gpd
from shapely.geometry import Polygon

out_path, lon, lat = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])

gdf = gpd.GeoDataFrame(
    {"id": [1, 2]},
    geometry=[
        Polygon([(lon - 0.4, lat - 0.4), (lon + 0.1, lat - 0.4), (lon + 0.1, lat + 0.1), (lon - 0.4, lat + 0.1)]),
        Polygon([(lon + 0.2, lat + 0.2), (lon + 0.6, lat + 0.2), (lon + 0.6, lat + 0.6), (lon + 0.2, lat + 0.6)]),
    ],
    crs="EPSG:4326",
)

tmpdir = tempfile.mkdtemp()
shp_path = os.path.join(tmpdir, "land_polygons.shp")
gdf.to_file(shp_path, driver="ESRI Shapefile")

os.makedirs(os.path.dirname(out_path), exist_ok=True)
with zipfile.ZipFile(out_path, "w") as zf:
    for ext in ("shp", "shx", "dbf", "prj", "cpg"):
        component = os.path.join(tmpdir, f"land_polygons.{ext}")
        if os.path.exists(component):
            zf.write(component, arcname=f"land_polygons.{ext}")
shutil.rmtree(tmpdir)
print(f"wrote synthetic fixture -> {out_path}")
PYEOF
fi

# --- REQUIRED: seed a CROPPED copy of the shared VIIRS-derived target
# geobox cache, centered on real land -- see validate-hard-gate-gadm.sh's
# header comment for why cropping is centered rather than grid-midpoint.
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
    echo "  Run 'pipeline run --source eog_viirs --step grid' once for real first,"
    echo "  or copy a viirs_geobox.pkl from wherever GRID has run before."
    exit 2
fi

# --- self-contained test config: isolated data_root, no HPC remote ---------
# Two source blocks for the SAME on-disk data: `misc` (OLD's nested-subsource
# shape) and `osm` (NEW's standalone-package shape; url/name defaults
# already match OSM_URL/RAW_REL so they're omitted here) -- same two-block
# pattern validate-hard-gate-gadm.sh uses for GADM.
cat > "$TEST_CONFIG" <<EOF
paths:
  data_root: "${TEST_ROOT}"
  local_index_dir: "${TEST_ROOT}/hpc_data_index"
remote:
  ssh_target: ""
  key_file: ""
sources:
  misc:
    type: "misc"
    data_path: "misc"
    sources:
      osm:
        url: "${OSM_URL}"
        name: "land-polygons-complete-4326.zip"
        subfolder: "osm"
  osm:
    type: "osm"
EOF

run_old() {
    local stage="$1"
    echo "$(date -Is): OLD  preprocess run --source misc --subsource osm --stage $stage"
    "$PYTHON_BIN" run.py preprocess run --config "$TEST_CONFIG" --source misc --subsource osm \
        --stage "$stage" --override \
        --dask-threads "$SLURM_CPUS_PER_TASK" --dask-memory-limit 4GiB \
        --temp-dir "${TEST_ROOT}/dask_tmp"
}

run_new() {
    local step="$1"
    echo "$(date -Is): NEW  pipeline run --source osm --step $step"
    "$PYTHON_BIN" run.py pipeline run --config "$TEST_CONFIG" --source osm \
        --step "$step" --override \
        --dask-threads "$SLURM_CPUS_PER_TASK" --dask-memory-limit 4GiB \
        --temp-dir "${TEST_ROOT}/dask_tmp"
}

# --- 1. OLD code produces the reference artefacts ---------------------------
run_old vector
run_old spatial

PREPARE_DIR="${TEST_ROOT}/misc/processed/stage_1/osm"
GRID_DIR="${TEST_ROOT}/misc/processed/stage_2/osm"
PREPARE_OLD_REF="${PREPARE_DIR}.old_reference"
GRID_OLD_REF="${GRID_DIR}.old_reference"

echo "$(date -Is): snapshotting OLD output before NEW code overwrites the same paths"
cp -a "$PREPARE_DIR" "$PREPARE_OLD_REF"
cp -a "$GRID_DIR" "$GRID_OLD_REF"

# --- 2. NEW code overwrites the same target paths ---------------------------
run_new prepare
run_new grid

# --- 3. Compare ---------------------------------------------------------------
echo "$(date -Is): comparing PREPARE output"
PREPARE_STATUS=0
"$PYTHON_BIN" scripts/compare_step_output.py \
    "${PREPARE_OLD_REF}/land_polygons_simplified.gpkg" "${PREPARE_DIR}/land_polygons_simplified.gpkg" || PREPARE_STATUS=$?

echo "$(date -Is): comparing GRID output"
GRID_STATUS=0
"$PYTHON_BIN" scripts/compare_step_output.py \
    "${GRID_OLD_REF}/land_mask.zarr" "${GRID_DIR}/land_mask.zarr" || GRID_STATUS=$?

echo "=============================================================="
echo "HARD-GATE PILOT RESULT -- source=osm window_px=$WINDOW_PX lon=$LON lat=$LAT"
echo "  PREPARE (vector  vs prepare): $([ $PREPARE_STATUS -eq 0 ] && echo EQUIVALENT || echo NOT_EQUIVALENT)"
echo "  GRID    (spatial vs grid):    $([ $GRID_STATUS -eq 0 ] && echo EQUIVALENT || echo NOT_EQUIVALENT)"
echo "  Test root (not auto-deleted, inspect or clean up manually): $TEST_ROOT"
echo "=============================================================="

if [ $PREPARE_STATUS -ne 0 ] || [ $GRID_STATUS -ne 0 ]; then
    exit 1
fi
exit 0
