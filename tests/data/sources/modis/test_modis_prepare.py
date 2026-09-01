"""ModisSource._execute_prepare wiring onto the shared run_tiled_prepare
driver: per-source-tile reproject + first-wins overlay
(`src.data.common.prepare.sinusoidal_mosaic`), target_geobox threading, and a
real tiled parquet write."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import yaml
from odc.geo.geobox import GeoBox
from rasterio.transform import from_bounds

from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources.modis.source import ModisSource
from src.data.sources.steps import PipelineStep, TargetSelection, marker_path


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


def _patch_geobox(monkeypatch, geobox):
    import src.data.common.geobox as geobox_module

    monkeypatch.setattr(geobox_module, "get_or_create_canonical_geobox", lambda cache_path: geobox)


def test_execute_prepare_writes_real_tiled_parquet_output(tmp_path, monkeypatch):
    source, ctx = _make_source(tmp_path, year_range=[2019, 2019])
    _write_tile_tif(
        os.path.join(source.output_root(PipelineStep.FETCH), "2019", "h18v04.tif"),
        290.0, ["lst_night_mean"],
    )

    _patch_geobox(monkeypatch, GeoBox.from_bbox((-1, -1, 1, 1), crs="EPSG:4326", resolution=0.5))  # 4x4, 2x2 tiles @ size 2
    source.tile_size = 2

    targets = source.plan(PipelineStep.PREPARE, TargetSelection())
    assert len(targets) == 1
    target = targets[0]

    assert source._execute_prepare(target) is True
    assert os.path.exists(marker_path(target.output_path))

    parts = sorted(Path(target.output_path).glob("ix=*/iy=*/part-*.parquet"))
    assert len(parts) == 4  # 2x2 tile grid x 1 year
    df = pd.concat(pd.read_parquet(p) for p in parts)
    assert set(df["year"].unique()) == {2019}
    assert set(df.columns) == {"cell_id", "year", "lst_night_mean"}
    assert (df["lst_night_mean"] == 290.0).all()


def test_execute_prepare_overlays_adjacent_source_tiles(tmp_path, monkeypatch):
    """Two genuinely-adjacent source tiles whose independently-pinned pixel
    grids overlap by a sliver used to crash `xr.combine_by_coords` with
    "duplicate values" (docs/design/15-modis-prepare-2002-tile-failures.md).
    Each source tile is now reprojected onto the output grid *individually*
    and overlaid, so sub-pixel misalignment between source tiles can never
    collide -- the exaggerated sliver overlap here just exercises the
    first-wins overlay seam."""
    source, ctx = _make_source(tmp_path, year_range=[2019, 2019])
    _write_tile_tif(
        os.path.join(source.output_root(PipelineStep.FETCH), "2019", "h18v04.tif"),
        1.0, ["lst_night_mean"], bounds=(-1, -1, 0.1, 1),
    )
    _write_tile_tif(
        os.path.join(source.output_root(PipelineStep.FETCH), "2019", "h19v04.tif"),
        2.0, ["lst_night_mean"], bounds=(-0.1, -1, 1, 1),
    )

    _patch_geobox(monkeypatch, GeoBox.from_bbox((-1, -1, 1, 1), crs="EPSG:4326", resolution=0.5))
    source.tile_size = 2

    target = source.plan(PipelineStep.PREPARE, TargetSelection())[0]
    assert source._execute_prepare(target) is True

    parts = sorted(Path(target.output_path).glob("ix=*/iy=*/part-*.parquet"))
    assert len(parts) == 4
    df = pd.concat(pd.read_parquet(p) for p in parts)
    vals = df["lst_night_mean"].values
    assert np.all(np.isfinite(vals))
    assert set(np.unique(vals)) == {1.0, 2.0}  # both source tiles land, no gaps


def test_execute_prepare_overlapping_source_tiles_first_wins(tmp_path, monkeypatch):
    """Where two source tiles cover the same output pixel, the
    lexicographically-first tile wins (deterministic `sorted(os.listdir)`
    order + all-NaN canvas + `combine_first`)."""
    source, ctx = _make_source(tmp_path, year_range=[2019, 2019])
    fetch_root = source.output_root(PipelineStep.FETCH)
    _write_tile_tif(os.path.join(fetch_root, "2019", "h18v04.tif"), 1.0, ["lst_night_mean"])  # full grid
    _write_tile_tif(os.path.join(fetch_root, "2019", "h19v04.tif"), 2.0, ["lst_night_mean"])  # full grid

    _patch_geobox(monkeypatch, GeoBox.from_bbox((-1, -1, 1, 1), crs="EPSG:4326", resolution=0.5))
    source.tile_size = 2

    target = source.plan(PipelineStep.PREPARE, TargetSelection())[0]
    assert source._execute_prepare(target) is True

    df = pd.concat(pd.read_parquet(p) for p in Path(target.output_path).glob("ix=*/iy=*/part-*.parquet"))
    assert set(np.unique(df["lst_night_mean"].values)) == {1.0}  # h18v04 < h19v04


def test_execute_prepare_partial_coverage_writes_nan_tiles_and_completes(tmp_path, monkeypatch):
    """An output tile with no overlapping source tile for the year is a
    legitimate state, not a failure: `raw_getter` returns a georegistered
    all-NaN dataset on `tile.geobox`, the unit is recorded `complete`, and
    the run still marks the output done
    (docs/design/15-modis-prepare-2002-tile-failures.md B1 -- previously
    crashed `xr_reproject` with "Can not reproject non-georegistered array")."""
    source, ctx = _make_source(tmp_path, year_range=[2019, 2019])
    # Grid big enough that the antimeridian-clamp on the padded selection
    # bbox actually restricts it (grid wider than 2*margin), so a source
    # tile confined to the NE corner does not reach the SW output tiles.
    _write_tile_tif(
        os.path.join(source.output_root(PipelineStep.FETCH), "2019", "h20v05.tif"),
        290.0, ["lst_night_mean"], bounds=(20, 20, 40, 40),
    )
    _patch_geobox(monkeypatch, GeoBox.from_bbox((-40, -40, 40, 40), crs="EPSG:4326", resolution=1.0))
    source.tile_size = 20  # 4x4 tiles

    target = source.plan(PipelineStep.PREPARE, TargetSelection())[0]
    assert source._execute_prepare(target) is True
    assert os.path.exists(marker_path(target.output_path))

    parts = sorted(Path(target.output_path).glob("ix=*/iy=*/part-*.parquet"))
    assert len(parts) == 16
    per_tile_allnan = [
        bool(np.isnan(pd.read_parquet(p)["lst_night_mean"].values).all()) for p in parts
    ]
    assert any(per_tile_allnan)  # the SW tiles have no coverage
    assert not all(per_tile_allnan)  # the NE tile(s) do


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
    `sinusoidal_mosaic`'s `raw_getter` uses, without running a full (slow)
    PREPARE pass -- docs/design/13-prepare-memory-parallelism.md.
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


def test_execute_prepare_processes_each_year_independently(tmp_path, monkeypatch):
    source, ctx = _make_source(tmp_path, year_range=[2019, 2020])
    _write_tile_tif(
        os.path.join(source.output_root(PipelineStep.FETCH), "2019", "h18v04.tif"), 1.0, ["lst_night_mean"]
    )
    _write_tile_tif(
        os.path.join(source.output_root(PipelineStep.FETCH), "2020", "h18v04.tif"), 2.0, ["lst_night_mean"]
    )

    _patch_geobox(monkeypatch, GeoBox.from_bbox((-1, -1, 1, 1), crs="EPSG:4326", resolution=0.5))
    source.tile_size = 2

    target = source.plan(PipelineStep.PREPARE, TargetSelection())[0]
    assert target.meta["years"] == [2019, 2020]
    assert source._execute_prepare(target) is True

    df = pd.concat(pd.read_parquet(p) for p in Path(target.output_path).glob("ix=*/iy=*/part-*.parquet"))
    assert set(df.loc[df["year"] == 2019, "lst_night_mean"].unique()) == {1.0}
    assert set(df.loc[df["year"] == 2020, "lst_night_mean"].unique()) == {2.0}
