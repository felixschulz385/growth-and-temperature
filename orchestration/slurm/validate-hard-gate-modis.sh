#!/bin/bash
#SBATCH --job-name=validate-hard-gate-modis
#SBATCH --output=./log/maintenance/hard_gate/%x-%j.out
#SBATCH --error=./log/maintenance/hard_gate/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --qos=6hours
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Hard-gate pilot run: docs/design/09-integrated-pipeline.md step 9's
# validation gate, "MODIS streaming" archetype, source=modis.
#
# SCOPE, deliberately narrower than the acag/gadm/berman_mining pilots --
# read before trusting a PASS here as full equivalence:
#
# PREPARE (src/data/preprocess/sources/modis.py stage=annual / src/data/
# sources/modis/source.py step=prepare) streams directly from Microsoft
# Planetary Computer's STAC catalog -- no local raw-file read path exists at
# all in either OLD or NEW, and there is no cached/mocked STAC response or
# real GeoTIFF fixture anywhere in this repo to substitute (checked). A
# compute node without outbound internet to planetarycomputer.microsoft.com
# + Azure blob storage cannot run PREPARE, full stop -- this script does NOT
# attempt it. PREPARE's target *paths* are already covered by the existing
# plan()-level oracle tests (tests/data/sources/modis/test_modis_plan.py,
# tests/data/preprocess/sources/test_characterization_modis.py); its actual
# *streamed bytes* are not verified equivalent by execution anywhere,
# including here. If this node happens to have PC network access, that is a
# separate, still-open pilot -- flagged, not silently assumed safe.
#
# This script instead stages a small SYNTHETIC multi-band GeoTIFF shaped
# exactly like `_write_annual_geotiff`'s real output (right CRS -- MODIS
# Sinusoidal -- right band-naming/filtering convention) directly at GRID's
# expected PREPARE-output input path, then validates GRID
# (mosaic-and-reproject) equivalence only. GRID's reprojection is the same
# `SpatialProcessor.create_empty_target_zarr`/`write_year_to_zarr` engine
# validate-hard-gate-acag.sh already exercises (confirmed by reading both
# call sites directly) -- MODIS's `_execute_grid` just calls it one layer
# more directly, with `resampling="nearest"` hardcoded rather than
# variable-driven, plus a MODIS-only pre-step (`_mosaic_tiles`, a plain
# `xr.combine_by_coords` over already-co-registered sinusoidal tiles, no
# reprojection).
#
# Geobox: unlike acag/gadm/berman_mining (which crop a pre-existing
# production `viirs_geobox.pkl`), MODIS's GRID builds its own canonical
# EPSG:6933 geobox via `get_or_create_canonical_geobox()`, cached at
# `&lt;data_root&gt;/canonical_geobox.pkl` -- a different cache, at the
# data_root's top level, confirmed identical in both OLD (`modis.py`) and
# NEW (`source.py::output_root`, which forces `grid_id="ease6933"`
# regardless of the repo-wide `pipeline.grid` config switch -- the one
# pre-existing, deliberate MODIS-only special case docs/design/05-migration.md
# §1 describes). Building one from scratch is pure PROJ math, no data
# dependency (docs/design/01-grid.md) -- both OLD and NEW call
# `get_or_create_canonical_geobox(cache_path)` with no resolution/lat-clip
# args, so whatever is cached at that path is used as-is. This script
# pre-seeds a small, coarse one directly -- no $DATA_NOBACKUP dependency at
# all for this pilot, unlike every other hard-gate script so far.
#
# Usage:
#   sbatch orchestration/slurm/validate-hard-gate-modis.sh [YEAR] [TILE] [RESOLUTION_M] [LAT_CLIP_DEG]
#   (YEAR defaults to 2020, TILE defaults to h18v08 -- near the equator/prime
#   meridian, chosen so it overlaps a small default-centered canonical grid.
#   RESOLUTION_M/LAT_CLIP_DEG default to 10000/15 -- deliberately coarse for
#   pilot speed, NOT the real 1000m/60deg production parameters.)

set -euo pipefail

PROJECT_ROOT="/scicore/home/meiera/schulz0022/projects/growth-and-temperature"
PYTHON_BIN="/scicore/home/meiera/schulz0022/miniforge-pypy3/envs/src/bin/python"
YEAR="${1:-2020}"
TILE="${2:-h18v08}"
RESOLUTION_M="${3:-10000}"
LAT_CLIP_DEG="${4:-15}"
PRODUCT="21A2"

cd "$PROJECT_ROOT"
mkdir -p "./log/maintenance/hard_gate"

LOG_FILE="./log/maintenance/hard_gate/validate-hard-gate-modis-${SLURM_JOB_ID}.log"
exec > >(tee -a "$LOG_FILE")
exec 2>&1

eval "$(/scicore/home/meiera/schulz0022/miniforge-pypy3/bin/conda shell.bash hook)"
conda activate src

TEST_ROOT="/scratch/schulz0022/hard_gate_modis_${SLURM_JOB_ID}"
TEST_CONFIG="${TEST_ROOT}/config.yaml"

echo "$(date -Is): Hard-gate pilot -- source=modis (GRID only) year=$YEAR tile=$TILE resolution_m=$RESOLUTION_M lat_clip_deg=$LAT_CLIP_DEG"
echo "Job ID:     $SLURM_JOB_ID"
echo "Test root:  $TEST_ROOT  (isolated -- production data_nobackup is never written to)"
echo "Log file:   $LOG_FILE"

STAGE1_DIR="${TEST_ROOT}/modis/${PRODUCT}/processed/stage_1/${YEAR}"
mkdir -p "$STAGE1_DIR"

# --- REQUIRED: stage a synthetic annual GeoTIFF at PREPARE's output path,
# shaped like _write_annual_geotiff's real output (see header comment) ------
echo "$(date -Is): generating synthetic annual GeoTIFF (no PC network access needed)"
"$PYTHON_BIN" - "$TILE" "${STAGE1_DIR}/${TILE}.tif" <<'PYEOF'
import sys

import numpy as np
import rasterio
from affine import Affine

from src.data.sources.modis.tiles import SINUSOIDAL_PROJ4, tile_bounds_m

tile_id, out_path = sys.argv[1], sys.argv[2]
h, v = int(tile_id[1:3]), int(tile_id[4:6])
x0, y0, x1, y1 = tile_bounds_m(h, v)

size = 60  # small synthetic tile
transform = Affine.translation(x0, y1) * Affine.scale((x1 - x0) / size, -(y1 - y0) / size)

rng = np.random.default_rng(0)
lst = rng.uniform(260.0, 300.0, size=(size, size)).astype("float32")
valid_count = rng.integers(1, 30, size=(size, size)).astype("float32")

with rasterio.open(
    out_path, "w", driver="GTiff", height=size, width=size, count=2, dtype="float32",
    crs=SINUSOIDAL_PROJ4, transform=transform, nodata=np.nan,
    compress="deflate", predictor=3, tiled=True,
) as dst:
    dst.write(lst, 1)
    dst.write(valid_count, 2)
    dst.set_band_description(1, "lst_night")
    dst.set_band_description(2, "valid_period_count_annual")

print(f"wrote synthetic MODIS tile {tile_id} -> {out_path} (bounds_m={x0},{y0},{x1},{y1})")
PYEOF

# --- REQUIRED: seed a small canonical EPSG:6933 geobox (no prod dependency,
# see header comment) --------------------------------------------------------
"$PYTHON_BIN" - "${TEST_ROOT}/canonical_geobox.pkl" "$RESOLUTION_M" "$LAT_CLIP_DEG" <<'PYEOF'
import pickle
import sys

from src.data.common.geobox.canonical import canonical_ease_geobox

out_path, resolution_m, lat_clip_deg = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
gbox = canonical_ease_geobox(resolution_m=resolution_m, lat_clip_deg=lat_clip_deg)
with open(out_path, "wb") as f:
    pickle.dump(gbox, f)
print(f"built canonical geobox: shape={gbox.shape} resolution_m={resolution_m} lat_clip_deg={lat_clip_deg}")
PYEOF

# --- self-contained test config: isolated data_root, no HPC remote ---------
cat > "$TEST_CONFIG" <<EOF
paths:
  data_root: "${TEST_ROOT}"
  local_index_dir: "${TEST_ROOT}/hpc_data_index"
remote:
  ssh_target: ""
  key_file: ""
sources:
  modis:
    type: "modis"
    data_path: "modis/${PRODUCT}"
    product: "${PRODUCT}"
    platform: "aqua"
    tiles: ["${TILE}"]
    year_range: [${YEAR}, ${YEAR}]
EOF

run_old() {
    echo "$(date -Is): OLD  preprocess run --source modis --stage spatial"
    "$PYTHON_BIN" run.py preprocess run --config "$TEST_CONFIG" --source modis \
        --stage spatial --override \
        --dask-threads "$SLURM_CPUS_PER_TASK" --dask-memory-limit 4GiB \
        --temp-dir "${TEST_ROOT}/dask_tmp"
}

run_new() {
    echo "$(date -Is): NEW  pipeline run --source modis --step grid"
    "$PYTHON_BIN" run.py pipeline run --config "$TEST_CONFIG" --source modis \
        --step grid --override \
        --dask-threads "$SLURM_CPUS_PER_TASK" --dask-memory-limit 4GiB \
        --temp-dir "${TEST_ROOT}/dask_tmp"
}

# --- 1. OLD code produces the reference artefact ----------------------------
run_old

GRID_DIR="${TEST_ROOT}/modis/${PRODUCT}/processed/stage_2_ease6933"
GRID_OLD_REF="${GRID_DIR}.old_reference"

echo "$(date -Is): snapshotting OLD output before NEW code overwrites the same paths"
cp -a "$GRID_DIR" "$GRID_OLD_REF"

# --- 2. NEW code overwrites the same target paths ---------------------------
run_new

# --- 3. Compare ---------------------------------------------------------------
echo "$(date -Is): comparing GRID output"
GRID_STATUS=0
"$PYTHON_BIN" scripts/compare_step_output.py \
    "${GRID_OLD_REF}/modis_${PRODUCT}_timeseries_reprojected.zarr" \
    "${GRID_DIR}/modis_${PRODUCT}_timeseries_reprojected.zarr" || GRID_STATUS=$?

echo "=============================================================="
echo "HARD-GATE PILOT RESULT -- source=modis (GRID only) year=$YEAR tile=$TILE"
echo "  GRID (spatial vs grid): $([ $GRID_STATUS -eq 0 ] && echo EQUIVALENT || echo NOT_EQUIVALENT)"
echo "  PREPARE: NOT exercised -- needs live Planetary Computer network access, see header comment"
echo "  Test root (not auto-deleted, inspect or clean up manually): $TEST_ROOT"
echo "=============================================================="

if [ $GRID_STATUS -ne 0 ]; then
    exit 1
fi
exit 0
