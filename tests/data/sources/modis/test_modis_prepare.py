"""ModisSource._execute_prepare wiring onto the shared run_tiled_prepare
driver: year-major mosaic memoization, target_geobox threading, and a real
tiled parquet write."""

import os

import numpy as np
import pytest
import rasterio
import yaml
from odc.geo.geobox import GeoBox
from rasterio.transform import from_bounds

import xarray as xr

from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources.modis.source import ModisSource, _trim_edge_overlap
from src.data.sources.steps import Completion, PipelineStep, StepTarget, TargetSelection, marker_path


def _make_source(tmp_path, **raw):
    ctx = PipelineContext(data_root=str(tmp_path / "data_root"), local_index_dir=str(tmp_path / "index"))
    cfg = SourceConfig.from_dict("modis", raw)
    return ModisSource(ctx, cfg), ctx


def _write_tile_tif(path, value, band_names, bounds=(-1, -1, 1, 1), size=8):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    transform = from_bounds(*bounds, size, size)
    with rasterio.open(
        path, "w", driver="GTiff", height=size, width=size, count=len(band_names),
        dtype="float32", crs="EPSG:4326", transform=transform, nodata=np.nan,
    ) as dst:
        for i, name in enumerate(band_names, start=1):
            dst.write(np.full((size, size), value, dtype="float32"), i)
            dst.set_band_description(i, name)


def test_execute_prepare_writes_real_tiled_parquet_output(tmp_path, monkeypatch):
    source, ctx = _make_source(tmp_path, year_range=[2019, 2019])
    _write_tile_tif(
        os.path.join(source.output_root(PipelineStep.FETCH), "2019", "h18v04.tif"),
        290.0, ["lst_night_mean"],
    )

    fake_geobox = GeoBox.from_bbox((-1, -1, 1, 1), crs="EPSG:4326", resolution=0.5)  # 4x4, 2x2 tiles @ size 2
    import src.data.common.geobox as geobox_module

    monkeypatch.setattr(geobox_module, "get_or_create_canonical_geobox", lambda cache_path: fake_geobox)
    source.tile_size = 2

    targets = source.plan(PipelineStep.PREPARE, TargetSelection())
    assert len(targets) == 1
    target = targets[0]

    assert source._execute_prepare(target) is True
    assert os.path.exists(marker_path(target.output_path))

    import pandas as pd
    from pathlib import Path

    parts = sorted(Path(target.output_path).glob("ix=*/iy=*/part-*.parquet"))
    assert len(parts) == 4  # 2x2 tile grid x 1 year
    df = pd.concat(pd.read_parquet(p) for p in parts)
    assert set(df["year"].unique()) == {2019}
    assert set(df.columns) == {"cell_id", "year", "lst_night_mean"}
    assert (df["lst_night_mean"] == 290.0).all()


def test_execute_prepare_handles_slightly_misaligned_adjacent_tiles(tmp_path, monkeypatch):
    """Real production failure (2026-08-25): MODIS FETCH pins `crs=`/
    `resolution=` per tile but not an explicit shared geobox, so two
    genuinely-adjacent tiles' pixel grids can end up a fraction of a pixel
    out of alignment -- `xr.combine_by_coords` requires exact
    non-overlapping coordinate labels and raised "duplicate values" the
    moment it met this. `rioxarray.merge` (now used instead) merges by
    actual georeferencing and tolerates it
    (docs/design/13-prepare-memory-parallelism.md)."""
    source, ctx = _make_source(tmp_path, year_range=[2019, 2019])
    # Two tiles whose nominal bounds overlap by a sliver (unlike real
    # adjacent-but-misaligned tiles, deliberately exaggerated here so an
    # 8x8-pixel fixture still reproduces a genuine coordinate collision).
    _write_tile_tif(
        os.path.join(source.output_root(PipelineStep.FETCH), "2019", "h18v04.tif"),
        1.0, ["lst_night_mean"], bounds=(-1, -1, 0.1, 1),
    )
    _write_tile_tif(
        os.path.join(source.output_root(PipelineStep.FETCH), "2019", "h19v04.tif"),
        2.0, ["lst_night_mean"], bounds=(-0.1, -1, 1, 1),
    )

    fake_geobox = GeoBox.from_bbox((-1, -1, 1, 1), crs="EPSG:4326", resolution=0.5)  # 4x4, 2x2 tiles @ size 2
    import src.data.common.geobox as geobox_module

    monkeypatch.setattr(geobox_module, "get_or_create_canonical_geobox", lambda cache_path: fake_geobox)
    source.tile_size = 2

    targets = source.plan(PipelineStep.PREPARE, TargetSelection())
    target = targets[0]

    assert source._execute_prepare(target) is True

    import pandas as pd
    from pathlib import Path

    parts = sorted(Path(target.output_path).glob("ix=*/iy=*/part-*.parquet"))
    assert len(parts) == 4
    df = pd.concat(pd.read_parquet(p) for p in parts)
    assert np.all(np.isfinite(df["lst_night_mean"].values))


def test_clamped_bbox_avoids_antimeridian_wrap_at_real_grid_corner_tile():
    """Real production failure (2026-08-25): `GeoBox.from_bbox` pixel-snaps
    a requested bbox's edges slightly *outside* the mathematically valid
    domain of a periodic (longitude-wrapping) CRS like EASE6933 -- the real
    canonical grid's own left edge sits ~470m past the true minimum x. Left
    unclamped, reprojecting a grid-corner tile's padded bbox to MODIS's
    sinusoidal CRS silently wraps to the opposite side of the world instead
    of erroring: confirmed live, tile `0000_0000`'s naive bbox matched
    104 of 282 fetched land tiles (spanning nearly the whole sinusoidal
    domain) instead of a geographically-plausible handful. This exercises
    the real canonical grid and the exact clamping arithmetic
    `ModisSource._execute_prepare`'s `raw_getter` uses, without running a
    full (slow) PREPARE pass -- docs/design/13-prepare-memory-parallelism.md.
    """
    from odc.geo.geom import box

    from src.data.common.geobox.canonical import canonical_ease_geobox
    from src.data.common import tiling
    from src.data.sources.modis import tiles as modis_util

    target_geobox = canonical_ease_geobox()
    tile = list(tiling.iter_tiles(target_geobox, tile_size=2048))[0]
    assert tile.id == "0000_0000"

    padded_bbox = tile.geobox.pad(32, 32).extent.boundingbox
    naive_sinu = tile.geobox.pad(32, 32).extent.to_crs(modis_util.SINUSOIDAL_PROJ4).boundingbox
    naive_width_km = (naive_sinu.right - naive_sinu.left) / 1000
    assert naive_width_km > 20_000  # reproduces the bug: spans most of the sinusoidal domain

    grid_bbox = target_geobox.extent.boundingbox
    margin = 2 * abs(target_geobox.resolution.x)
    clamped_left = max(padded_bbox.left, grid_bbox.left + margin)
    clamped_bottom = max(padded_bbox.bottom, grid_bbox.bottom + margin)
    clamped_right = min(padded_bbox.right, grid_bbox.right - margin)
    clamped_top = min(padded_bbox.top, grid_bbox.top - margin)
    assert clamped_left < clamped_right and clamped_bottom < clamped_top

    clamped = box(clamped_left, clamped_bottom, clamped_right, clamped_top, crs=tile.geobox.crs)
    fixed_sinu = clamped.to_crs(modis_util.SINUSOIDAL_PROJ4).boundingbox
    fixed_width_km = (fixed_sinu.right - fixed_sinu.left) / 1000
    assert fixed_width_km < 10_000  # a geographically plausible width for one output tile

    with open("orchestration/configs/data.yaml") as f:
        land_tiles = yaml.safe_load(f)["sources"]["modis"]["land_tiles"]
    overlap_count = sum(
        1
        for t in land_tiles
        for h, v in [(int(t[1:3]), int(t[4:6]))]
        for x0, y0, x1, y1 in [modis_util.tile_bounds_m(h, v)]
        if x1 >= fixed_sinu.left and x0 <= fixed_sinu.right and y1 >= fixed_sinu.bottom and y0 <= fixed_sinu.top
    )
    assert overlap_count < 30  # was 104/282 before the fix


def test_trim_edge_overlap_resolves_real_multi_pixel_boundary_overlap():
    """Real production failure (2026-08-26): after pixel-snapping fixes
    sub-pixel misalignment, `xr.combine_by_coords` still crashed with
    "duplicate values" combining 37 real tiles for tile 0000_0001/year
    2002 -- genuine multi-pixel overlap between adjacent h/v tiles
    sharing a boundary column, which snapping alone can't fix.
    `_trim_edge_overlap` trims that shared full-height boundary strip off
    the later tile before combining (docs/design/13-prepare-memory-parallelism.md)."""

    def make_tile(x_vals, y_vals, value):
        data = np.full((len(y_vals), len(x_vals)), value, dtype=np.float32)
        return xr.Dataset({"v": (("y", "x"), data)}, coords={"x": x_vals, "y": y_vals})

    y_vals = np.arange(5, dtype=float)
    # Three tiles in a row sharing full-height 2-column boundary overlaps
    # with their neighbours -- the standard MODIS/GLASS-MODIS h/v grid
    # overlap pattern, not a partial corner overlap.
    tile0 = make_tile(np.arange(0, 10, dtype=float), y_vals, 0.0)
    tile1 = make_tile(np.arange(8, 18, dtype=float), y_vals, 1.0)
    tile2 = make_tile(np.arange(16, 26, dtype=float), y_vals, 2.0)

    trimmed = _trim_edge_overlap([tile0, tile1, tile2])
    merged = xr.combine_by_coords(trimmed, combine_attrs="drop_conflicts", join="outer")

    assert not merged["x"].to_index().has_duplicates
    assert not merged["y"].to_index().has_duplicates
    assert merged.sizes["x"] == 26  # 0..25 inclusive, no gaps or double-counted columns
    assert merged.sizes["y"] == 5
    assert np.isfinite(merged["v"].values).all()


def test_execute_prepare_reuses_one_years_mosaic_at_a_time(tmp_path, monkeypatch):
    source, ctx = _make_source(tmp_path, year_range=[2019, 2020])
    _write_tile_tif(
        os.path.join(source.output_root(PipelineStep.FETCH), "2019", "h18v04.tif"), 1.0, ["lst_night_mean"]
    )
    _write_tile_tif(
        os.path.join(source.output_root(PipelineStep.FETCH), "2020", "h18v04.tif"), 2.0, ["lst_night_mean"]
    )

    fake_geobox = GeoBox.from_bbox((-1, -1, 1, 1), crs="EPSG:4326", resolution=0.5)
    import src.data.common.geobox as geobox_module

    monkeypatch.setattr(geobox_module, "get_or_create_canonical_geobox", lambda cache_path: fake_geobox)
    source.tile_size = 2

    targets = source.plan(PipelineStep.PREPARE, TargetSelection())
    target = targets[0]
    assert target.meta["years"] == [2019, 2020]

    assert source._execute_prepare(target) is True

    import pandas as pd
    from pathlib import Path

    parts = sorted(Path(target.output_path).glob("ix=*/iy=*/part-*.parquet"))
    df = pd.concat(pd.read_parquet(p) for p in parts)
    assert set(df.loc[df["year"] == 2019, "lst_night_mean"].unique()) == {1.0}
    assert set(df.loc[df["year"] == 2020, "lst_night_mean"].unique()) == {2.0}
