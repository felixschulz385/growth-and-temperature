"""
Backbone raster-pipeline validation (docs/design/05-migration.md §5 step 8).

Exercises the new EASE-Grid 2.0 (EPSG:6933) pipeline end-to-end on a small
subset: canonical grid construction -> latitude-band kernel registry ->
mask-aware disc convolution -> Zarr disc-ladder store -> ring-mean
tabularization. Reports structural sanity checks (row counts match the mask,
no unexpected NaN/inf, disc counts within a plausible bound for each tile's
latitude band).

What this script does NOT do: run the econometric "sanity check against
expected coefficient behaviour" that docs/design/05-migration.md §5 step 8
also calls for. That needs `src/analysis/` wiring against a real assembled
panel, which is out of scope for the backbone implementation itself (hard
constraint: don't touch `src/analysis/` beyond the interface it consumes).
This script validates the raster pipeline that feeds that step, not the
estimation step itself.

Two modes:
  --variable-zarr PATH   Real mode. PATH is a Zarr store already regridded
                          onto the canonical EASE6933 grid -- i.e. the output
                          of `run.py data run --source X --step grid`
                          with `orchestration/configs/data.yaml`'s
                          `pipeline.grid` set to `ease6933`
                          (docs/design/09-integrated-pipeline.md §3; that
                          config key defaults to `legacy_4326`, so this must
                          be set explicitly before running `grid`). Must
                          expose `--data-var` on a `y`/`x` grid matching the
                          canonical GeoBox, optionally with a leading `time`
                          or `year` dimension selected via --years.

                          Validity mask: as of this script's writing, the
                          `grid` step (`SpatialProcessor` in
                          `src/data/common/raster/spatial.py`) does not emit
                          a separate boolean validity variable -- invalid
                          cells are nodata-filled within the data variable
                          itself (a `_FillValue`/`nodata` attribute, NaN by
                          default for float arrays). This script derives the
                          mask from that automatically; `--mask-var`, if
                          given and present in the store, overrides that and
                          is used directly instead (for any future/other
                          source that does emit a real mask variable).
  (omitted)               Synthetic mode. Generates a small in-memory field
                          with a validity mask (a few holes punched in) --
                          no real data required. Useful on its own for
                          confirming the pinned environment (environment.yml)
                          and this pipeline's logic actually run on a given
                          HPC node, independent of whether regridded source
                          data exists there yet.

Samples --n-tiles tiles spread evenly across the canonical grid's tiling
(not one contiguous region) so the run exercises multiple latitude bands of
the elliptical kernel registry, not just one -- a more informative structural
check than a single region, at the cost of not literally matching the design
doc's "one region" phrasing (that phrasing was about keeping the downstream
*econometric* check's subset small, which doesn't apply to a structural
raster check).

Usage:
  python scripts/validate_backbone_subset.py --hpc-root /path/to/hpc_root

  python scripts/validate_backbone_subset.py --hpc-root /path/to/hpc_root \\
      --variable-zarr /path/to/eog_ease6933.zarr \\
      --data-var DNB_BRDF_Corrected_NTL --years 2020 2021
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from odc.geo.geobox import GeoboxTiles
from odc.geo.xr import xr_zeros

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.assemble.ring_means import tabularize_tile  # noqa: E402
from src.data.common.geobox.canonical import get_or_create_canonical_geobox  # noqa: E402
from src.data.common.neighbourhood.constants import (  # noqa: E402
    DEFAULT_DISC_LADDER_KM,
    DEFAULT_R_MAX_KM,
)
from src.data.common.neighbourhood.discs import convolve_tile  # noqa: E402
from src.data.common.neighbourhood.kernels import (  # noqa: E402
    anisotropy_scales,
    compute_band_edges,
    get_or_create_registry,
)
from src.data.common.neighbourhood.store import (  # noqa: E402
    create_empty_disc_sum_store,
    create_empty_disc_count_store,
    write_disc_tile,
)


def setup_logging(log_file: str = None) -> logging.Logger:
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )
    return logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--hpc-root", required=True, help="HPC root for geobox/kernel-registry caching")
    parser.add_argument(
        "--variable-zarr", default=None, help="Real mode: path to a canonical-grid-aligned Zarr store"
    )
    parser.add_argument("--data-var", default="value")
    parser.add_argument(
        "--mask-var",
        default=None,
        help="Optional: name of an explicit boolean/0-1 validity variable in the store. "
        "If omitted or not present, the mask is derived from the data variable's own "
        "_FillValue/nodata attribute (or NaN, for float arrays with neither).",
    )
    parser.add_argument("--years", type=int, nargs="+", default=[2020])
    parser.add_argument("--tile-size", type=int, default=2048)
    parser.add_argument("--n-tiles", type=int, default=4, help="Tiles sampled across the grid's latitude range")
    parser.add_argument("--resolution-m", type=float, default=1000.0)
    parser.add_argument("--lat-clip-deg", type=float, default=60.0)
    parser.add_argument("--r-max-km", type=float, default=DEFAULT_R_MAX_KM)
    parser.add_argument("--ladder-km", type=float, nargs="+", default=DEFAULT_DISC_LADDER_KM)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write the validation disc-ladder stores (default: <hpc-root>/prepared/misc/misc/backbone_validate)",
    )
    parser.add_argument("--log-file", default=None)
    return parser.parse_args()


def _sample_tile_indices(tiles: GeoboxTiles, n_tiles: int) -> list:
    """Row/col pairs spread evenly across the tile grid's latitude range."""
    n_rows, n_cols = tiles.shape[0], tiles.shape[1]
    total = n_rows * n_cols
    n_tiles = max(1, min(n_tiles, total))
    flat_indices = np.linspace(0, total - 1, n_tiles, dtype=int)
    return [(int(idx // n_cols), int(idx % n_cols)) for idx in flat_indices]


def build_synthetic_field(geobox, logger) -> tuple:
    """A whole-grid uniform field with a few holes punched in, entirely in-memory."""
    logger.info("SYNTHETIC MODE -- no --variable-zarr given; generating an in-memory test field")
    variable = xr_zeros(geobox, dtype="float32") + 5.0
    mask = xr_zeros(geobox, dtype="bool")
    mask.values[...] = True
    # punch a few holes so the mask-aware convolution's missing-data handling
    # (docs/design/03-neighbourhood-engine.md §2) is actually exercised
    rng = np.random.default_rng(0)
    ny, nx = geobox.shape
    for _ in range(max(1, (ny * nx) // 5000)):
        cy, cx = rng.integers(0, ny), rng.integers(0, nx)
        half = rng.integers(1, 5)
        mask.values[max(0, cy - half) : cy + half, max(0, cx - half) : cx + half] = False
    return variable, mask


def load_real_field(path: str, data_var: str, logger) -> tuple:
    logger.info("REAL MODE -- loading %s (data_var=%s)", path, data_var)
    ds = xr.open_zarr(path, consolidated=False, mask_and_scale=False)
    if data_var not in ds:
        raise ValueError(f"{data_var!r} not found in {path} (available: {list(ds.data_vars)})")
    return ds, ds[data_var]


def derive_mask(data: xr.DataArray, ds: xr.Dataset, mask_var: str, year: int, logger) -> xr.DataArray:
    """Validity mask for one year's slice of a grid-step output.

    The `grid` step (`SpatialProcessor` in `src/data/common/raster/spatial.py`)
    does not currently emit a separate boolean validity variable -- invalid
    cells are nodata-filled within the data variable itself. Prefer an
    explicit `mask_var`, if the caller named one and it's actually present
    (keeps this working for any future/other source that does emit a real
    mask); otherwise derive from `data`'s own `_FillValue`/`nodata` attribute,
    falling back to NaN for float arrays with neither.
    """
    if mask_var and mask_var in ds:
        mask_da = ds[mask_var]
        if "time" in mask_da.dims or "year" in mask_da.dims:
            mask_da = _select_year(mask_da, year, logger)
        logger.info("Using explicit mask variable %r", mask_var)
        return mask_da.astype(bool)

    fill_value = data.attrs.get("_FillValue", data.attrs.get("nodata"))
    if fill_value is not None:
        logger.info("Deriving mask from %s's _FillValue/nodata attr = %r", data.name, fill_value)
        if isinstance(fill_value, float) and np.isnan(fill_value):
            return ~np.isnan(data)
        return data != fill_value

    if np.issubdtype(data.dtype, np.floating):
        logger.warning(
            "No mask variable and no _FillValue/nodata attr on %s -- falling back to a NaN-based "
            "mask; verify this actually matches the store's nodata convention before trusting the "
            "result.",
            data.name,
        )
        return ~np.isnan(data)

    logger.warning(
        "No mask variable, no _FillValue/nodata attr, and %s is not a float dtype -- cannot "
        "determine validity from the data alone; treating every cell as valid. Pass --mask-var "
        "pointing at a real variable if this is wrong.",
        data.name,
    )
    return xr.ones_like(data, dtype=bool)


def _select_year(da: xr.DataArray, year: int, logger) -> xr.DataArray:
    for dim in ("time", "year"):
        if dim in da.dims:
            if dim == "time":
                candidates = [c for c in da.coords["time"].values if str(c)[:4] == str(year)]
                if not candidates:
                    raise ValueError(f"No time coordinate found for year {year}")
                return da.sel(time=candidates[0])
            return da.sel(year=year)
    logger.warning("No time/year dimension found on %s; using as-is for every requested year", da.name)
    return da


def validate_tile(
    variable: xr.DataArray,
    mask: xr.DataArray,
    tile_gbox,
    row: int,
    col: int,
    year: int,
    r_max_m: float,
    ladder_km,
    registry,
    disc_sum_path,
    disc_count_path,
    logger,
) -> dict:
    S_d, N_d = convolve_tile(variable, mask, tile_gbox, r_max_m, ladder_km, registry)
    write_disc_tile(disc_sum_path, S_d, year=year, variable="S_d")
    write_disc_tile(disc_count_path, N_d, year=year, variable="N_d")

    tile_mask = mask.odc.crop(tile_gbox.extent) if hasattr(mask, "odc") else mask
    df = tabularize_tile(S_d, N_d, tile_mask, tile_gbox, ix=col, iy=row, year=year)

    lat_deg = tile_gbox.geographic_extent.centroid.coords[0][1] if hasattr(tile_gbox, "geographic_extent") else None
    scale_ew, scale_ns = anisotropy_scales(lat_deg if lat_deg is not None else 0.0)
    max_radius_px = max(ladder_km) * 1000.0 / (tile_gbox.resolution.x)
    plausible_n_ceiling = np.pi * (max_radius_px * max(scale_ew, scale_ns)) ** 2 * 1.5  # 50% slack

    issues = []
    expected_rows = int(np.asarray(tile_mask.values).astype(bool).sum())
    if len(df) != expected_rows:
        issues.append(f"row count {len(df)} != valid-mask count {expected_rows}")

    for r in ladder_km:
        l_col, n_col = f"L_{r}km", f"N_{r}km"
        if not np.isfinite(df[l_col].replace([np.inf, -np.inf], np.nan).dropna()).all():
            issues.append(f"{l_col} contains inf")
        if (df[n_col] > plausible_n_ceiling).any():
            issues.append(f"{n_col} exceeds plausible ceiling ({plausible_n_ceiling:.0f})")

    result = {
        "row": row,
        "col": col,
        "year": year,
        "lat_deg": lat_deg,
        "n_rows": len(df),
        "expected_rows": expected_rows,
        "issues": issues,
    }
    for r in ladder_km:
        l_col = f"L_{r}km"
        nan_frac = df[l_col].isna().mean() if len(df) else float("nan")
        result[f"nan_frac_{r}km"] = nan_frac

    status = "OK" if not issues else "ISSUES"
    logger.info(
        "tile(row=%s,col=%s) year=%s lat=%s rows=%s/%s status=%s",
        row,
        col,
        year,
        f"{lat_deg:.1f}" if lat_deg is not None else "n/a",
        len(df),
        expected_rows,
        status,
    )
    if issues:
        for issue in issues:
            logger.warning("  - %s", issue)

    return result


def main() -> int:
    args = parse_args()
    logger = setup_logging(args.log_file)

    logger.info("=== Backbone raster-pipeline validation (docs/design/05-migration.md step 8) ===")
    logger.info("hpc_root=%s resolution_m=%s lat_clip_deg=%s r_max_km=%s ladder_km=%s", args.hpc_root, args.resolution_m, args.lat_clip_deg, args.r_max_km, args.ladder_km)

    cache_dir = Path(args.hpc_root) / "prepared" / "misc" / "misc"
    output_dir = Path(args.output_dir) if args.output_dir else cache_dir / "backbone_validate"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Building/loading canonical EASE6933 geobox...")
    geobox = get_or_create_canonical_geobox(
        cache_dir / "ease6933_geobox.pkl",
        resolution_m=args.resolution_m,
        lat_clip_deg=args.lat_clip_deg,
    )
    logger.info("Canonical geobox: shape=%s crs=%s", geobox.shape, geobox.crs)

    logger.info("Building/loading latitude-band elliptical kernel registry...")
    edges = compute_band_edges(lat_max_deg=args.lat_clip_deg)
    registry = get_or_create_registry(
        cache_dir / "kernel_registry.pkl",
        band_edges_deg=edges,
        radii_km=args.ladder_km,
        resolution_m=args.resolution_m,
    )
    logger.info("Kernel registry: %d bands x %d radii", registry.n_bands, len(args.ladder_km))

    real_ds = None
    if args.variable_zarr:
        real_ds, variable_full = load_real_field(args.variable_zarr, args.data_var, logger)
    else:
        variable_full, mask_full = build_synthetic_field(geobox, logger)

    tiles = GeoboxTiles(geobox, (args.tile_size, args.tile_size))
    tile_indices = _sample_tile_indices(tiles, args.n_tiles)
    logger.info("Sampling %d tile(s) across %d total tiles: %s", len(tile_indices), tiles.shape[0] * tiles.shape[1], tile_indices)

    r_max_m = args.r_max_km * 1000.0
    disc_sum_path = output_dir / "S_d_validation.zarr"
    disc_count_path = output_dir / "N_d_validation.zarr"
    create_empty_disc_sum_store(disc_sum_path, geobox, args.years, args.ladder_km, tile_size=args.tile_size)
    create_empty_disc_count_store(disc_count_path, geobox, args.years, args.ladder_km, tile_size=args.tile_size)

    all_results = []
    for year in args.years:
        if args.variable_zarr:
            variable_year = _select_year(variable_full, year, logger)
            mask_year = derive_mask(variable_year, real_ds, args.mask_var, year, logger)
        else:
            variable_year = variable_full
            mask_year = mask_full

        for row, col in tile_indices:
            tile_gbox = tiles[row, col]
            result = validate_tile(
                variable_year,
                mask_year,
                tile_gbox,
                row,
                col,
                year,
                r_max_m,
                args.ladder_km,
                registry,
                disc_sum_path,
                disc_count_path,
                logger,
            )
            all_results.append(result)

    n_total = len(all_results)
    n_with_issues = sum(1 for r in all_results if r["issues"])
    logger.info("=== Summary ===")
    logger.info("tiles x years validated: %d", n_total)
    logger.info("tiles with issues: %d", n_with_issues)
    for r in all_results:
        if r["issues"]:
            logger.info("  FAILED tile(row=%s,col=%s) year=%s: %s", r["row"], r["col"], r["year"], "; ".join(r["issues"]))

    if n_with_issues:
        logger.error("VALIDATION FAILED: %d/%d tile-years had issues (see above)", n_with_issues, n_total)
        return 1

    logger.info("VALIDATION PASSED: all %d tile-years structurally sound", n_total)
    logger.info(
        "Reminder: this covers the raster pipeline only. The econometric sanity check "
        "against expected coefficient behaviour (docs/design/05-migration.md step 8) still "
        "needs to be run separately once src/analysis/ is wired to this handoff schema."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
