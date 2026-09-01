"""ModisSource's FETCH step: completion is plain local-disk presence
(`Completion.PATH_EXISTS`, matching every other source,
`src.data.common.fetch.manifest`), and a failed (year, tile) unit's
retry/error history lives in a small JSON sidecar under FETCH's own output
root, written via `manifest.record_failure`/`clear_failure` directly (MODIS
has no crawl catalog, see module docstring).
"""

import numpy as np
import pandas as pd
import pytest
import rioxarray  # noqa: F401 -- registers the .rio accessor _write_annual_geotiff needs
import xarray as xr

from src.data.common import statusfile
from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources.modis.source import ModisSource
from src.data.sources.steps import PipelineStep, TargetSelection

pytest.importorskip("rasterio")


def _make_source(tmp_path, tiles=("h18v04",), year_range=(2019, 2019), source_id="modis"):
    data_root = str(tmp_path / "data_root")
    local_index_dir = str(tmp_path / "index")
    ctx = PipelineContext(data_root=data_root, local_index_dir=local_index_dir)
    cfg = SourceConfig.from_dict(source_id, {"year_range": list(year_range), "tiles": list(tiles)})
    return ModisSource(ctx, cfg), ctx


def _fake_dataset() -> xr.Dataset:
    lst = xr.DataArray(
        np.full((1, 2, 2), 280.0, dtype="float32"),
        dims=("time", "y", "x"),
        coords={"time": [pd.Timestamp("2019-06-01")]},
    )
    qc = xr.DataArray(np.zeros((1, 2, 2), dtype="uint8"), dims=("time", "y", "x"), coords={"time": lst.time})
    # Day LST/QC -- distinct value (300.0) from night's 280.0 so day/night
    # are visibly distinguishable in test assertions; "good" QC like night's.
    lst_day = xr.DataArray(np.full((1, 2, 2), 300.0, dtype="float32"), dims=("time", "y", "x"), coords={"time": lst.time})
    qc_day = xr.DataArray(np.zeros((1, 2, 2), dtype="uint8"), dims=("time", "y", "x"), coords={"time": lst.time})
    return xr.Dataset({"lst": lst, "qc": qc, "lst_day": lst_day, "qc_day": qc_day})


def _fake_dataset_with_one_corrupted_pixel() -> xr.Dataset:
    # Reproduces the real case found in tile h09v02/2002: a single-
    # observation pixel with a "good" QC flag but a decoded LST value
    # (900K) that's physically impossible -- QC bits alone don't reject it;
    # decode_qc_valid_mask()'s LST-range check does.
    #
    # mandatory_qa=00, error_bits=11 -- for ModisSource's default product
    # ("21A2"), bits 7&6 are *inverted* from "11A1" (increasing value =
    # better, not worse -- see decode_qc_valid_mask()'s module comment), so
    # 11 is 21A2's *best* category (<1K), not its worst.
    values = np.array([[280.0, 900.0], [280.0, 280.0]], dtype="float32")
    lst = xr.DataArray(values[np.newaxis], dims=("time", "y", "x"), coords={"time": [pd.Timestamp("2019-06-01")]})
    qc_byte = 0b11 << 6
    qc = xr.DataArray(np.full((1, 2, 2), qc_byte, dtype="uint8"), dims=("time", "y", "x"), coords={"time": lst.time})
    # Day LST/QC -- required by _execute_fetch's main-variant band check but
    # not what this test exercises (that's the night-side impossible-value
    # guard), so a plain "good" fixture like _fake_dataset()'s.
    lst_day = xr.DataArray(np.full((1, 2, 2), 300.0, dtype="float32"), dims=("time", "y", "x"), coords={"time": lst.time})
    qc_day = xr.DataArray(np.zeros((1, 2, 2), dtype="uint8"), dims=("time", "y", "x"), coords={"time": lst.time})
    return xr.Dataset({"lst": lst, "qc": qc, "lst_day": lst_day, "qc_day": qc_day})


def test_execute_fetch_writes_output_and_clears_any_prior_failure_status(tmp_path, monkeypatch):
    source, ctx = _make_source(tmp_path)
    monkeypatch.setattr(source, "_search_items", lambda tile, year: ["fake-item"])
    monkeypatch.setattr(source, "_load_tile_year", lambda items, tile: _fake_dataset())

    targets = source.plan(PipelineStep.FETCH, TargetSelection())
    assert len(targets) == 1
    target = targets[0]

    ok = source.execute(target)
    assert ok is True
    source.close()

    import os

    assert os.path.exists(target.output_path)
    status_dir = source.output_root(PipelineStep.FETCH)
    assert statusfile.read(statusfile.status_path(status_dir, target.key)) is None


def test_execute_fetch_excludes_a_physically_impossible_pixel_despite_good_qc(tmp_path, monkeypatch):
    import rasterio

    source, ctx = _make_source(tmp_path)
    monkeypatch.setattr(source, "_search_items", lambda tile, year: ["fake-item"])
    monkeypatch.setattr(source, "_load_tile_year", lambda items, tile: _fake_dataset_with_one_corrupted_pixel())

    targets = source.plan(PipelineStep.FETCH, TargetSelection())
    target = targets[0]
    assert source.execute(target) is True
    source.close()

    with rasterio.open(target.output_path) as src:
        band_index = list(src.descriptions).index("lst_night_mean") + 1
        lst_night = src.read(band_index)

    assert lst_night[0, 0] == pytest.approx(280.0)
    assert lst_night[1, 0] == pytest.approx(280.0)
    assert lst_night[1, 1] == pytest.approx(280.0)
    # The corrupted pixel (900K, single observation, "good" QC) is excluded
    # from the composite entirely -- not clamped, not averaged in -- since
    # it was the only observation for that pixel that period.
    assert np.isnan(lst_night[0, 1])


def _fake_dataset_with_extended_bands() -> xr.Dataset:
    ds = _fake_dataset()
    for var in ("emis_29", "emis_31", "emis_32", "view_angle", "view_time"):
        ds[var] = xr.DataArray(
            np.full((1, 2, 2), 0.97, dtype="float32"), dims=("time", "y", "x"), coords={"time": ds.time}
        )
    return ds


def test_main_variant_writes_lst_stats_and_valid_counts_only(tmp_path, monkeypatch):
    source, _ = _make_source(tmp_path, source_id="modis")
    assert source.variant == "main"
    monkeypatch.setattr(source, "_search_items", lambda tile, year: ["fake-item"])
    monkeypatch.setattr(source, "_load_tile_year", lambda items, tile: _fake_dataset_with_extended_bands())

    target = source.plan(PipelineStep.FETCH, TargetSelection())[0]
    assert source.execute(target) is True
    source.close()

    import rasterio

    with rasterio.open(target.output_path) as src:
        assert set(src.descriptions) == {
            "lst_night_mean", "lst_night_median", "lst_night_sd",
            "valid_period_count_annual", "valid_month_count_annual",
            "lst_day_mean", "lst_day_median", "lst_day_sd",
            "valid_period_count_day_annual", "valid_month_count_day_annual",
        }


def _fake_dataset_with_distinct_day_night_bad_pixels() -> xr.Dataset:
    # Night's bad pixel is (0, 1); day's is (1, 0) -- a different pixel, so
    # a test can prove the two QC decodes are genuinely independent instead
    # of one leaking into the other.
    night_values = np.array([[280.0, 900.0], [280.0, 280.0]], dtype="float32")
    day_values = np.array([[300.0, 300.0], [900.0, 300.0]], dtype="float32")
    lst = xr.DataArray(
        night_values[np.newaxis], dims=("time", "y", "x"), coords={"time": [pd.Timestamp("2019-06-01")]}
    )
    lst_day = xr.DataArray(day_values[np.newaxis], dims=("time", "y", "x"), coords={"time": lst.time})
    qc_byte = 0b11 << 6  # "good" mandatory QA, best error category (21A2)
    qc = xr.DataArray(np.full((1, 2, 2), qc_byte, dtype="uint8"), dims=("time", "y", "x"), coords={"time": lst.time})
    qc_day = xr.DataArray(np.full((1, 2, 2), qc_byte, dtype="uint8"), dims=("time", "y", "x"), coords={"time": lst.time})
    return xr.Dataset({"lst": lst, "qc": qc, "lst_day": lst_day, "qc_day": qc_day})


def test_day_and_night_valid_masks_are_computed_independently(tmp_path, monkeypatch):
    import rasterio

    source, _ = _make_source(tmp_path, source_id="modis")
    monkeypatch.setattr(source, "_search_items", lambda tile, year: ["fake-item"])
    monkeypatch.setattr(source, "_load_tile_year", lambda items, tile: _fake_dataset_with_distinct_day_night_bad_pixels())

    target = source.plan(PipelineStep.FETCH, TargetSelection())[0]
    assert source.execute(target) is True
    source.close()

    with rasterio.open(target.output_path) as src:
        night_band = list(src.descriptions).index("lst_night_mean") + 1
        day_band = list(src.descriptions).index("lst_day_mean") + 1
        lst_night = src.read(night_band)
        lst_day = src.read(day_band)

    # Night's impossible pixel (0, 1) is excluded from lst_night_mean but
    # unaffected in lst_day_mean; day's impossible pixel (1, 0) is the
    # reverse -- each mask only excludes its own bad pixel.
    assert np.isnan(lst_night[0, 1])
    assert lst_night[1, 0] == pytest.approx(280.0)
    assert np.isnan(lst_day[1, 0])
    assert lst_day[0, 1] == pytest.approx(300.0)


def test_extended_variant_writes_emissivity_and_view_bands_only(tmp_path, monkeypatch):
    source, _ = _make_source(tmp_path, source_id="modis_extended")
    assert source.variant == "extended"
    monkeypatch.setattr(source, "_search_items", lambda tile, year: ["fake-item"])
    monkeypatch.setattr(source, "_load_tile_year", lambda items, tile: _fake_dataset_with_extended_bands())

    target = source.plan(PipelineStep.FETCH, TargetSelection())[0]
    assert source.execute(target) is True
    source.close()

    import rasterio

    with rasterio.open(target.output_path) as src:
        assert set(src.descriptions) == {"emis_29", "emis_31", "emis_32", "view_angle", "view_time"}


def test_main_and_extended_variants_use_distinct_paths(tmp_path):
    main_source, _ = _make_source(tmp_path, source_id="modis")
    extended_source, _ = _make_source(tmp_path, source_id="modis_extended")

    assert main_source.cfg.data_path != extended_source.cfg.data_path
    assert main_source.output_root(PipelineStep.FETCH) != extended_source.output_root(PipelineStep.FETCH)
    assert main_source._prepare_output_path() != extended_source._prepare_output_path()


def test_execute_fetch_records_failure_status_on_no_stac_items(tmp_path, monkeypatch):
    source, ctx = _make_source(tmp_path)
    monkeypatch.setattr(source, "_search_items", lambda tile, year: [])

    targets = source.plan(PipelineStep.FETCH, TargetSelection())
    target = targets[0]

    ok = source.execute(target)
    assert ok is False
    source.close()

    status_dir = source.output_root(PipelineStep.FETCH)
    status = statusfile.read(statusfile.status_path(status_dir, target.key))
    assert status is not None
    assert status["status"] == "retrying"
    assert status["attempts"] == 1


def test_execute_fetch_retried_after_failure_can_succeed(tmp_path, monkeypatch):
    """Partial-download resumability: a tile-year that failed once (e.g. a
    transient STAC search error) is retried on the next `data run` call
    and, on success, ends up complete -- not stuck failed forever."""
    source, ctx = _make_source(tmp_path)
    monkeypatch.setattr(source, "_search_items", lambda tile, year: [])

    targets = source.plan(PipelineStep.FETCH, TargetSelection())
    target = targets[0]
    assert source.execute(target) is False
    source.close()

    source2, _ = _make_source(tmp_path)
    monkeypatch.setattr(source2, "_search_items", lambda tile, year: ["fake-item"])
    monkeypatch.setattr(source2, "_load_tile_year", lambda items, tile: _fake_dataset())
    targets2 = source2.plan(PipelineStep.FETCH, TargetSelection())
    assert source2.execute(targets2[0]) is True
    source2.close()

    import os

    assert os.path.exists(target.output_path)
    status_dir = source.output_root(PipelineStep.FETCH)
    # Cleared by the successful retry, not left stuck 'retrying'.
    assert statusfile.read(statusfile.status_path(status_dir, target.key)) is None


class _FakeStacItem:
    def __init__(self, item_id, platform="aqua", h=None, v=None):
        self.id = item_id
        self.properties = {"platform": platform}
        if h is not None:
            self.properties["modis:horizontal-tile"] = h
        if v is not None:
            self.properties["modis:vertical-tile"] = v


def test_search_items_filters_out_neighbouring_tiles(tmp_path, monkeypatch):
    """`_search_items` asks the server for exactly one tile's items via a
    CQL2 `filter` on `modis:horizontal-tile`/`modis:vertical-tile`
    (confirmed live against Planetary Computer, 2026-08-26: 92 items in
    0.8s vs. 4,692 items in 12.1s for the equivalent bbox search, verified
    identical once platform-filtered) -- but this client-side filter stays
    as defense-in-depth against a backend that doesn't honor `filter`, or
    items whose tile properties are missing entirely (the `platform`
    property has shown exactly that gap before, see the tripwire above this
    filter in `_search_items`). Drops non-matching-tile items using the
    STAC-reported `modis:horizontal-tile`/`modis:vertical-tile` properties,
    falling back to parsing the `h##v##` segment out of `item.id` for items
    missing those properties, and keeping an item outright if neither is
    available (docs/design/13-prepare-memory-parallelism.md)."""
    source, ctx = _make_source(tmp_path, tiles=("h08v05",))

    matching = _FakeStacItem("MYD21A2.A2019001.h08v05.061.2019010000000", h=8, v=5)
    neighbour = _FakeStacItem("MYD21A2.A2019001.h09v05.061.2019010000000", h=9, v=5)
    id_only_matching = _FakeStacItem("MYD21A2.A2019001.h08v05.061.2019010000001")
    id_only_neighbour = _FakeStacItem("MYD21A2.A2019001.h07v05.061.2019010000001")
    no_tile_info = _FakeStacItem("some-other-id-format")

    class _FakeSearch:
        def items(self):
            return [matching, neighbour, id_only_matching, id_only_neighbour, no_tile_info]

    class _FakeClient:
        def search(self, **kwargs):
            return _FakeSearch()

    monkeypatch.setattr(source, "_get_stac_client", lambda: _FakeClient())

    result = source._search_items("h08v05", 2019)
    assert result == [matching, id_only_matching, no_tile_info]


def test_search_items_queries_by_cql2_tile_filter_not_bbox(tmp_path, monkeypatch):
    """The actual STAC query should ask the server for this tile via a CQL2
    `filter` on the collection's queryable `modis:horizontal-tile`/
    `modis:vertical-tile` properties -- not a `bbox` (the old bbox approach
    had to fall back to a full-globe search for any tile whose footprint
    crosses the antimeridian, confirmed live against Planetary Computer,
    2026-08-26)."""
    source, ctx = _make_source(tmp_path, tiles=("h08v05",))

    captured = {}

    class _FakeSearch:
        def items(self):
            return []

    class _FakeClient:
        def search(self, **kwargs):
            captured.update(kwargs)
            return _FakeSearch()

    monkeypatch.setattr(source, "_get_stac_client", lambda: _FakeClient())

    source._search_items("h08v05", 2019)

    assert "bbox" not in captured
    assert captured["filter_lang"] == "cql2-json"
    assert captured["filter"] == {
        "op": "and",
        "args": [
            {"op": "=", "args": [{"property": "modis:horizontal-tile"}, 8]},
            {"op": "=", "args": [{"property": "modis:vertical-tile"}, 5]},
        ],
    }


def test_search_items_retries_transient_api_error_and_succeeds(tmp_path, monkeypatch):
    """Planetary Computer intermittently returns a transient `APIError`
    (observed: "The request exceeded the maximum allowed time") -- a single
    hiccup must not permanently fail the (year, tile) target."""
    from pystac_client.exceptions import APIError

    source, ctx = _make_source(tmp_path, tiles=("h08v05",))
    matching = _FakeStacItem("MYD21A2.A2019001.h08v05.061.2019010000000", h=8, v=5)

    calls = {"n": 0}

    class _FakeSearch:
        def items(self):
            calls["n"] += 1
            if calls["n"] < 3:
                raise APIError("The request exceeded the maximum allowed time, please try again.")
            return [matching]

    class _FakeClient:
        def search(self, **kwargs):
            return _FakeSearch()

    monkeypatch.setattr(source, "_get_stac_client", lambda: _FakeClient())
    monkeypatch.setattr("src.data.sources.modis.source.time.sleep", lambda seconds: None)

    result = source._search_items("h08v05", 2019)

    assert result == [matching]
    assert calls["n"] == 3


def test_search_items_raises_after_exhausting_retries(tmp_path, monkeypatch):
    from pystac_client.exceptions import APIError

    source, ctx = _make_source(tmp_path, tiles=("h08v05",))

    class _FakeSearch:
        def items(self):
            raise APIError("The request exceeded the maximum allowed time, please try again.")

    class _FakeClient:
        def search(self, **kwargs):
            return _FakeSearch()

    monkeypatch.setattr(source, "_get_stac_client", lambda: _FakeClient())
    monkeypatch.setattr("src.data.sources.modis.source.time.sleep", lambda seconds: None)

    with pytest.raises(APIError):
        source._search_items("h08v05", 2019, max_retries=3)
