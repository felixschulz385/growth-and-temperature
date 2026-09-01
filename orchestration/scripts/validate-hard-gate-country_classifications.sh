#!/bin/bash
#SBATCH --job-name=validate-hard-gate-country_classifications
#SBATCH --output=./log/maintenance/hard_gate/%x-%j.out
#SBATCH --error=./log/maintenance/hard_gate/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --qos=6hours
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Hard-gate pilot run: docs/design/09-integrated-pipeline.md step 9's
# validation gate, "tabular-join" archetype, source=country_classifications
# (UNDP HDI + World Bank income group, merged on iso3, rasterized onto
# GADM's country-id grid).
#
# What this proves: that OLD code (src/data/preprocess/sources/misc.py,
# subsource=country_classifications, stage=vector/spatial) and NEW code
# (src/data/sources/misc/country_classifications.py, step=prepare/grid)
# produce equivalent parquet (PREPARE) and Zarr (GRID) output for the same
# input, via scripts/compare_step_output.py.
#
# Real, confirmed-by-reading-both-codepaths discrepancy exercised here, NOT
# fixed (out of this pilot's scope, flagged for whoever does the step-10
# cutover): OLD dumps BOTH the HDI and World Bank raw files into
# misc/raw/hdi/ (hardcoded "hdi" subfolder in
# src/data/preprocess/sources/misc.py, applies to both files -- not even
# reading the per-file `subfolder` config key); NEW puts both into one
# shared misc/raw/country_classifications/ folder. This script stages TWO
# copies of each raw file, one at each location, so both codepaths find
# their inputs from the same run.
#
# Also exercises NEW's `REQUIRES = (("gadm", PipelineStep.GRID),)` -- unlike
# plad (REQUIRES on gadm's PREPARE), this needs GADM's *rasterized* grid
# output, not just its simplified vector files, confirmed by reading both
# OLD's _generate_spatial_targets (which requires
# misc/processed/stage_2/gadm/countries_grid.zarr to already exist, silently
# empty-plans otherwise) and NEW's _plan_grid (same path, via
# layout.output_root). REQUIRES is enforced by src/cli/data/handlers.py
# ::_check_requires for EVERY step, including `prepare` -- so this script
# builds a real GADM GRID output first via GadmSource directly, before
# touching country_classifications at all, not just before its own GRID
# step.
#
# CONFIRMED ON REAL SLURM, WAIVED, NOT A BUG IN THIS SCRIPT: OLD's PREPARE
# step (src/data/preprocess/sources/misc.py's HDI branch,
# `hdi.loc[:, "year"] = hdi["year"].str[4:].astype(int)`) raises
# `TypeError: Invalid value ... for dtype 'str'` under this HPC's pandas
# (pyarrow-string-backed, >=3.0) -- this repo pins no pandas version
# (docs/design/05-migration.md §3), so OLD cannot execute this step at all
# on this node; no execution-based diff is possible here without a
# throwaway legacy-pandas environment. NEW's src/data/sources/misc/hdi.py
# already has this fixed (plain bracket assignment instead of
# `.loc[:, ...] =`, with a code comment explaining exactly why) -- confirmed
# NEW's PREPARE+GRID run cleanly start to finish under pandas 3.0.2 in
# isolation. Decision (see docs/design/09-integrated-pipeline.md §14): the
# code-level comparison above is accepted as sufficient sign-off for this
# one step -- OLD is deleted in step 10 regardless -- so this pilot's
# tabular-join archetype coverage rests on code review for PREPARE's
# HDI-parsing specifically, not on this script's execution.
#
# Safety: everything below runs against an ISOLATED scratch data_root, never
# against $DATA_NOBACKUP -- same pattern as validate-hard-gate-acag.sh.
#
# Data required: none from production. Builds a small synthetic GADM
# fixture (ADM_0 only, two countries) plus small synthetic HDI/World Bank
# fixtures matching the exact shapes tests/data/sources/misc/
# test_hdi_worldbank.py already proves parse correctly (same iso3 codes as
# the GADM fixture, so the final classification grid has real content
# rather than being vacuously all-False).
#
# Usage:
#   sbatch orchestration/slurm/validate-hard-gate-country_classifications.sh [WINDOW_PX] [LON] [LAT]
#   (WINDOW_PX defaults to 300, LON/LAT default to 10.0/50.0 -- central
#   Europe, same default as validate-hard-gate-gadm.sh.)

set -euo pipefail

PROJECT_ROOT="/scicore/home/meiera/schulz0022/projects/growth-and-temperature"
PYTHON_BIN="/scicore/home/meiera/schulz0022/miniforge-pypy3/envs/gnt/bin/python"
WINDOW_PX="${1:-300}"
LON="${2:-10.0}"
LAT="${3:-50.0}"

cd "$PROJECT_ROOT"
mkdir -p "./log/maintenance/hard_gate"

LOG_FILE="./log/maintenance/hard_gate/validate-hard-gate-country_classifications-${SLURM_JOB_ID}.log"
exec > >(tee -a "$LOG_FILE")
exec 2>&1

eval "$(/scicore/home/meiera/schulz0022/miniforge-pypy3/bin/conda shell.bash hook)"
conda activate gnt

DATA_NOBACKUP="${DATA_NOBACKUP:-$PROJECT_ROOT/data_nobackup}"
TEST_ROOT="/scratch/schulz0022/hard_gate_cc_${SLURM_JOB_ID}"
TEST_CONFIG="${TEST_ROOT}/config.yaml"

echo "$(date -Is): Hard-gate pilot -- source=country_classifications window_px=$WINDOW_PX lon=$LON lat=$LAT"
echo "Job ID:     $SLURM_JOB_ID"
echo "Test root:  $TEST_ROOT  (isolated -- production data_nobackup is never written to)"
echo "Log file:   $LOG_FILE"

mkdir -p "${TEST_ROOT}/hpc_data_index" \
         "${TEST_ROOT}/misc/raw/hdi" \
         "${TEST_ROOT}/misc/raw/country_classifications" \
         "${TEST_ROOT}/misc/processed/stage_1/misc"

# --- REQUIRED: seed a CROPPED copy of the shared VIIRS-derived target
# geobox cache, centered on real content -- see validate-hard-gate-gadm.sh's
# header comment for why cropping is centered rather than grid-midpoint.
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

# --- REQUIRED prerequisite: a real GADM GRID output (see header comment on
# REQUIRES=(gadm,GRID)) -- built directly via GadmSource here rather than
# shelling out to validate-hard-gate-gadm.sh, to keep this script
# self-contained and because country_classifications only needs the ADM_0
# (country) rasterization, not GADM's own PREPARE/GRID equivalence (already
# covered by validate-hard-gate-gadm.sh).
#
# This driver is a real .py file, not a `python - <<EOF` stdin heredoc,
# because GadmSource.execute() spins up a real dask LocalCluster/Client for
# rasterization -- a stdin-fed script has no on-disk path for
# multiprocessing's spawned workers to re-exec, which fails with
# `FileNotFoundError: .../<stdin>` and loops retrying indefinitely (found by
# actually running this pilot on SLURM, not assumed).
GADM_PREREQ_SCRIPT="${TEST_ROOT}/build_gadm_prereq.py"
cat > "$GADM_PREREQ_SCRIPT" <<'PYEOF'
import os
import sys

import geopandas as gpd
from shapely.geometry import MultiPolygon, box


def main():
    test_root, lon, lat = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])

    from src.data.pipeline.context import PipelineContext
    from src.data.pipeline.config import SourceConfig
    from src.data.sources.misc.gadm import GadmSource
    from src.data.sources.steps import PipelineStep, TargetSelection

    # Minimal ADM_0-only GADM fixture: two countries side by side.
    adm0 = gpd.GeoDataFrame(
        [
            {"GID_0": "AAA", "geometry": MultiPolygon([box(lon - 0.6, lat - 0.3, lon - 0.1, lat + 0.3)])},
            {"GID_0": "BBB", "geometry": MultiPolygon([box(lon + 0.1, lat - 0.3, lon + 0.6, lat + 0.3)])},
        ],
        crs="EPSG:4326",
    )
    prepare_dir = os.path.join(test_root, "misc", "processed", "stage_1", "gadm")
    os.makedirs(prepare_dir, exist_ok=True)
    adm0.to_file(os.path.join(prepare_dir, "gadm_levelADM_0_simplified.gpkg"), driver="GPKG")

    ctx = PipelineContext(data_root=test_root, local_index_dir=os.path.join(test_root, "hpc_data_index"), dask_threads=2)
    cfg = SourceConfig(source_id="gadm", data_path="misc", namespace="gadm", override=True)
    source = GadmSource(ctx, cfg)

    grid_targets = source.plan(PipelineStep.GRID, TargetSelection())
    for t in grid_targets:
        ok = source.execute(t)
        print(f"GADM GRID prerequisite -> {ok}")
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
PYEOF
# PYTHONPATH is required here: unlike a `python -` stdin heredoc (where
# sys.path[0] is '' -- i.e. resolved against cwd, which `cd "$PROJECT_ROOT"`
# above already set), a real on-disk .py file gets its OWN directory
# ($TEST_ROOT) as sys.path[0], which has no `src` package -- confirmed by
# actually hitting `ModuleNotFoundError: No module named 'src'` running this
# locally before handing it off.
PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "$PYTHON_BIN" "$GADM_PREREQ_SCRIPT" "$TEST_ROOT" "$LON" "$LAT"

# --- REQUIRED: synthetic HDI/World Bank raw fixtures, iso3 matching the
# GADM fixture above (AAA, BBB) -- exact shapes proven to parse correctly by
# tests/data/sources/misc/test_hdi_worldbank.py, not guessed.
echo "$(date -Is): generating synthetic HDI/World Bank fixtures"
"$PYTHON_BIN" - "${TEST_ROOT}/misc/raw/hdi/HDR25.csv" "${TEST_ROOT}/misc/raw/hdi/DR0095334.xlsx" <<'PYEOF'
import sys
import pandas as pd

hdi_path, wb_path = sys.argv[1], sys.argv[2]

# HDI: Low <0.550, Medium <0.700, High <0.800, else Very High.
cols = {"iso3": ["AAA", "BBB"]}
for year in range(1990, 2024):
    cols[f"hdi_{year}"] = [0.500, 0.750]  # AAA=Low, BBB=High
pd.DataFrame(cols).to_csv(hdi_path, index=False, encoding="latin1")
print(f"wrote {hdi_path}")

header_row = ["Code", "Country", "FY91", "FY99", "FY11"]
rows = [["pad"] * 5 for _ in range(4)]
rows.append(header_row)
for _ in range(6):
    rows.append(["ZZZ", "Padding", "..", "..", ".."])
rows.append(["AAA", "Country A", "L", "L", "LM"])
rows.append(["BBB", "Country B", "H", "H", "H"])
rows.append(["trailer1", "", "", "", ""])
rows.append(["trailer2", "", "", "", ""])
pd.DataFrame(rows).to_excel(wb_path, sheet_name="Country Analytical History", header=False, index=False)
print(f"wrote {wb_path}")
PYEOF
# Second copy at NEW's shared raw path (see header comment on the OLD/NEW
# raw-folder discrepancy this script deliberately works around, not fixes).
cp "${TEST_ROOT}/misc/raw/hdi/HDR25.csv" "${TEST_ROOT}/misc/raw/country_classifications/HDR25.csv"
cp "${TEST_ROOT}/misc/raw/hdi/DR0095334.xlsx" "${TEST_ROOT}/misc/raw/country_classifications/DR0095334.xlsx"

# --- REQUIRED (OLD only): parquet index -- NEW's PREPARE checks raw-file
# existence directly and does not consult this.
"$PYTHON_BIN" - "${TEST_ROOT}/hpc_data_index/parquet_misc.parquet" <<'PYEOF'
import sys
import pandas as pd

out_path = sys.argv[1]
pd.DataFrame(
    [
        {"relative_path": "hdi/HDR25.csv", "status_category": "completed"},
        {"relative_path": "hdi/DR0095334.xlsx", "status_category": "completed"},
    ]
).to_parquet(out_path)
print(f"wrote index -> {out_path}")
PYEOF

# --- self-contained test config: isolated data_root, no HPC remote ---------
# `misc.sources.{hdi,wb}` values are cosmetic for OLD here (only consulted
# by the FETCH-side MiscDataSource, never by PREPARE/GRID -- confirmed by
# reading get_preprocessing_targets directly) but must exist for
# MiscPreprocessor.__init__ not to raise. `gadm:` and `country_classifications:`
# blocks are for NEW; data_path/namespace on gadm must be explicit (same
# REQUIRES footgun as validate-hard-gate-plad.sh).
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
      hdi:
        url: "https://example.invalid/HDR25.csv"
        name: "HDR25.csv"
        subfolder: "hdi"
      wb:
        url: "https://example.invalid/DR0095334.xlsx"
        name: "DR0095334.xlsx"
        subfolder: "worldbank"
  gadm:
    type: "gadm"
    data_path: "misc"
    namespace: "gadm"
  country_classifications:
    type: "country_classifications"
    data_path: "misc"
    namespace: "country_classifications"
    hdi_name: "HDR25.csv"
    worldbank_name: "DR0095334.xlsx"
EOF

run_old() {
    local stage="$1"
    echo "$(date -Is): OLD  preprocess run --source misc --subsource country_classifications --stage $stage"
    "$PYTHON_BIN" run.py preprocess run --config "$TEST_CONFIG" --source misc --subsource country_classifications \
        --stage "$stage" --override \
        --dask-threads "$SLURM_CPUS_PER_TASK" --dask-memory-limit 4GiB \
        --temp-dir "${TEST_ROOT}/dask_tmp"
}

run_new() {
    local step="$1"
    echo "$(date -Is): NEW  data run --source country_classifications --step $step"
    "$PYTHON_BIN" run.py data run --config "$TEST_CONFIG" --source country_classifications \
        --step "$step" --override \
        --dask-threads "$SLURM_CPUS_PER_TASK" --dask-memory-limit 4GiB \
        --temp-dir "${TEST_ROOT}/dask_tmp"
}

# --- 1. OLD code produces the reference artefacts ---------------------------
run_old vector
run_old spatial

PREPARE_DIR="${TEST_ROOT}/misc/processed/stage_1/country_classifications"
GRID_DIR="${TEST_ROOT}/misc/processed/stage_2/country_classifications"
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
    "${PREPARE_OLD_REF}/classifications.parquet" "${PREPARE_DIR}/classifications.parquet" || PREPARE_STATUS=$?

echo "$(date -Is): comparing GRID output"
GRID_STATUS=0
"$PYTHON_BIN" scripts/compare_step_output.py \
    "${GRID_OLD_REF}/classifications_grid.zarr" "${GRID_DIR}/classifications_grid.zarr" || GRID_STATUS=$?

echo "=============================================================="
echo "HARD-GATE PILOT RESULT -- source=country_classifications window_px=$WINDOW_PX lon=$LON lat=$LAT"
echo "  PREPARE (vector  vs prepare): $([ $PREPARE_STATUS -eq 0 ] && echo EQUIVALENT || echo NOT_EQUIVALENT)"
echo "  GRID    (spatial vs grid):    $([ $GRID_STATUS -eq 0 ] && echo EQUIVALENT || echo NOT_EQUIVALENT)"
echo "  Test root (not auto-deleted, inspect or clean up manually): $TEST_ROOT"
echo "=============================================================="

if [ $PREPARE_STATUS -ne 0 ] || [ $GRID_STATUS -ne 0 ]; then
    exit 1
fi
exit 0
