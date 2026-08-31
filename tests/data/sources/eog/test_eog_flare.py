"""EogFlareSource: filename->year mapping, PREPARE planning, and the
distance-banded rasterization (`_rasterize_tile`)."""

import os

import numpy as np
import pytest
from odc.geo.geobox import GeoBox

from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources.eog.flare import EogFlareSource
from src.data.sources.steps import Completion, PipelineStep, StepTarget, TargetSelection


def _make_source(tmp_path, grid_id="ease6933", **raw):
    ctx = PipelineContext(
        data_root=str(tmp_path / "data_root"), local_index_dir=str(tmp_path / "index"), grid_id=grid_id
    )
    cfg = SourceConfig.from_dict("eog_flare", {"data_path": "eog/flare", **raw})
    return EogFlareSource(ctx, cfg), ctx


def _write_raw(source, filename):
    raw_root = source.output_root(PipelineStep.FETCH)
    os.makedirs(raw_root, exist_ok=True)
    open(os.path.join(raw_root, filename), "w").close()


_COMBINED = "VIIRS_Global_flaring_d.7_slope_0.029353_2012-2016_v20221211_web.xlsx"


def _annual(year):
    return f"VIIRS_Global_flaring_d.7_slope_0.029353_{year}_v20230208_web.xlsx"


# --- filename -> year(s) -------------------------------------------------

def test_years_for_filename_expands_combined_span():
    assert EogFlareSource._years_for_filename(_COMBINED) == [2012, 2013, 2014, 2015, 2016]


def test_years_for_filename_single_year():
    assert EogFlareSource._years_for_filename(_annual(2021)) == [2021]


def test_list_remote_files_year_filter_picks_the_covering_file(tmp_path, monkeypatch):
    source, _ = _make_source(tmp_path)
    monkeypatch.setattr(
        source, "_scrape_xlsx_links",
        lambda: [(_COMBINED, "http://x/" + _COMBINED), (_annual(2021), "http://x/" + _annual(2021))],
    )
    assert source.list_remote_files({"year": 2014})[0][0] == _COMBINED
    assert source.list_remote_files({"year": 2021})[0][0] == _annual(2021)
    assert source.list_remote_files({"year": 2019}) == []


def test_get_all_entrypoints_covers_config_year_range(tmp_path):
    source, _ = _make_source(tmp_path, year_range=[2012, 2021])
    assert source.get_all_entrypoints() == [{"year": y} for y in range(2012, 2022)]


# --- PREPARE planning --------------------------------------------------

def test_prepare_plan_broadcasts_combined_file_to_each_year(tmp_path):
    source, _ = _make_source(tmp_path)
    _write_raw(source, _COMBINED)
    _write_raw(source, _annual(2017))

    target = source.plan(PipelineStep.PREPARE, TargetSelection())[0]
    assert target.meta["years"] == [2012, 2013, 2014, 2015, 2016, 2017]
    # 2012-2016 all resolve to the one combined file; 2017 to its own.
    assert target.meta["raw_files"][2013] == _COMBINED
    assert target.meta["raw_files"][2016] == _COMBINED
    assert target.meta["raw_files"][2017] == _annual(2017)
    # one input per distinct file
    assert set(target.inputs) == {_COMBINED, _annual(2017)}
    assert target.meta["expected_vars"] == ("flare_band",)
    assert target.meta["value_range"] == (0, 3)
    assert target.meta["sparse_vars"] == ("flare_band",)


def test_prepare_plan_empty_without_raw_files(tmp_path):
    source, _ = _make_source(tmp_path)
    assert source.plan(PipelineStep.PREPARE, TargetSelection()) == []


# --- rasterization ---------------------------------------------------

def _first_tile(geobox, tile_size):
    from src.data.common import tiling

    return next(iter(tiling.iter_tiles(geobox, tile_size=tile_size)))


def test_rasterize_tile_bands_by_distance_from_flare_point(tmp_path, monkeypatch):
    source, _ = _make_source(tmp_path)
    # ~1.1 km pixels around (lon=10, lat=0); 40x40 grid.
    geobox = GeoBox.from_bbox((9.8, -0.2, 10.2, 0.2), crs="EPSG:4326", resolution=0.01)
    tile = _first_tile(geobox, tile_size=64)  # single tile covers the whole grid

    monkeypatch.setattr(source, "_load_flare_points", lambda _p: np.array([[10.0, 0.0]]))
    monkeypatch.setattr(source, "_resolve_source_file_path", lambda p: p)

    ds = source._rasterize_tile(tile, 2020, {2020: "dummy.xlsx"})
    band = ds["flare_band"].values
    assert band.dtype == np.uint8
    assert set(np.unique(band)).issubset({0, 1, 2, 3})

    h, w = band.shape
    cy, cx = h // 2, w // 2
    assert band[cy, cx] == 3  # flare's own pixel
    assert band[cy, cx + 1] == 2  # ~1 km away -> within 2 km ring
    assert band[cy, cx + 3] == 1  # ~3.3 km away -> within 5 km ring only
    assert band[0, 0] == 0  # tile corner, ~15 km away


def test_rasterize_tile_all_zero_when_no_flares_near(tmp_path, monkeypatch):
    source, _ = _make_source(tmp_path)
    geobox = GeoBox.from_bbox((9.8, -0.2, 10.2, 0.2), crs="EPSG:4326", resolution=0.01)
    tile = _first_tile(geobox, tile_size=64)

    # flare far outside the tile + halo
    monkeypatch.setattr(source, "_load_flare_points", lambda _p: np.array([[50.0, 50.0]]))
    monkeypatch.setattr(source, "_resolve_source_file_path", lambda p: p)

    band = source._rasterize_tile(tile, 2020, {2020: "dummy.xlsx"})["flare_band"].values
    assert band.dtype == np.uint8
    assert not band.any()
