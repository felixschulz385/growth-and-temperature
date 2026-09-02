"""GlassModisSource's FETCH step (docs/design/12-glass-modis-rebuild.md §4):
one `(tile, year)` target per land tile x year (unlike the old per-(year,
day[, tile]) crawl -- see test_glass_avhrr_fetch.py for that shape, still
used by GlassAvhrrSource). `_execute_fetch()` iterates the tile-year's daily
listing itself and combines matched `.hdf` downloads into one annual
GeoTIFF; the listing-match/`.hdf`-only-filter logic is exercised directly
here.
"""

import os

import numpy as np
import pytest
import rioxarray as rxr
import xarray as xr

from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources.glass.modis import GlassModisSource
import src.data.sources.glass.modis as glass_modis_module
from src.data.sources.steps import Completion, PipelineStep, TargetSelection


def _make_source(tmp_path, source_id="glass_modis", **extra_raw):
    ctx = PipelineContext(data_root=str(tmp_path / "data_root"), local_index_dir=str(tmp_path / "index"))
    base_url = {
        "glass_modis": "https://glass.hku.hk/archive/LST/MODIS/Daily/1KM/",
        "glass_ta_modis": "https://glass.hku.hk/archive/Ta/MODIS/",
    }[source_id]
    day_range = extra_raw.pop("day_range", {"start": [2019, 1], "end": [2019, 3]})
    raw = {"base_url": base_url, "day_range": day_range, "land_tiles": ["h08v05", "h01v07"], **extra_raw}
    cfg = SourceConfig.from_dict(source_id, raw)
    return GlassModisSource(ctx, cfg), ctx


def test_variant_derived_from_registered_id(tmp_path):
    assert _make_source(tmp_path, "glass_modis")[0].variant == "lst"
    assert _make_source(tmp_path, "glass_ta_modis")[0].variant == "ta"
    assert _make_source(tmp_path, "glass_modis")[0].band_names == ("LST",)
    assert _make_source(tmp_path, "glass_ta_modis")[0].band_names == ("Ta_min", "Ta_mean", "Ta_max")


def test_plan_fetch_target_shape_is_tile_year(tmp_path):
    source, _ = _make_source(tmp_path, day_range={"start": [2019, 1], "end": [2020, 3]})
    targets = source.plan(PipelineStep.FETCH, TargetSelection())
    assert {t.key for t in targets} == {"2019/h08v05", "2019/h01v07", "2020/h08v05", "2020/h01v07"}
    for t in targets:
        assert t.meta["tile"] in ("h08v05", "h01v07")
        assert t.meta["year"] in (2019, 2020)
        assert t.completion == Completion.PATH_EXISTS
        assert t.output_path == os.path.join(
            source.output_root(PipelineStep.FETCH), str(t.meta["year"]), f"{t.meta['tile']}.tif"
        )


def test_listing_filters_to_hdf_only():
    # Ta's directory listing has multiple sidecar files per tile/day sharing
    # the same A{year}{day}.{tile}. token (doc §1) -- .hdf.xml/.jpg previews
    # must never match instead of the real .hdf data file.
    listing = [
        ("GLASS18A01.V10.A2000055.h03v06.2023300.hdf", "https://x/data.hdf"),
        ("GLASS18A01.V10.A2000055.h03v06.2023300.hdf.xml", "https://x/data.hdf.xml"),
        ("GLASS18A01.V10.A2000055.h03v06.2023300.jpg", "https://x/preview.jpg"),
    ]
    match = GlassModisSource._match_in_listing(listing, 2000, 55, "h03v06")
    assert match == ("GLASS18A01.V10.A2000055.h03v06.2023300.hdf", "https://x/data.hdf")


def test_match_in_listing_requires_tile():
    listing = [
        ("GLASS06A01.V01.A2020055.h00v08.2022021.hdf", "https://x/h00v08.hdf"),
        ("GLASS06A01.V01.A2020055.h01v07.2022021.hdf", "https://x/h01v07.hdf"),
    ]
    match = GlassModisSource._match_in_listing(listing, 2020, 55, "h01v07")
    assert match == ("GLASS06A01.V01.A2020055.h01v07.2022021.hdf", "https://x/h01v07.hdf")
    assert GlassModisSource._match_in_listing(listing, 2020, 55, "h02v02") is None


async def _fake_download_async(url, path, session=None):
    open(path, "wb").close()


def _fake_band(value: float) -> xr.DataArray:
    arr = xr.DataArray(
        np.full((2, 2), value, dtype="float64"),
        dims=("y", "x"),
        coords={"y": [1.0, 0.0], "x": [0.0, 1.0]},
    )
    return arr.rio.write_crs("EPSG:4326")


def _fake_multiband(values, band_dim_size, long_name=None):
    """A single DataArray with a `band` dim of size `band_dim_size` --
    what GDAL's HDF4-EOS driver actually returns for Ta in production
    (grouping same-shape SDS into one multi-band raster instead of a list
    of per-subdataset DataArrays, see `_open_hdf_bands`'s docstring)."""
    arr = xr.DataArray(
        np.stack([np.full((2, 2), v, dtype="float64") for v in values]),
        dims=("band", "y", "x"),
        coords={"band": list(range(1, band_dim_size + 1)), "y": [1.0, 0.0], "x": [0.0, 1.0]},
    )
    if long_name is not None:
        arr.attrs["long_name"] = long_name
    return arr.rio.write_crs("EPSG:4326")


def test_scale_bands_applies_scale_factor_and_prefers_attr():
    # `rxr.open_rasterio(masked=True)` does not apply scale_factor, so raw
    # int16 DN (~16000-37000 for Ta) is scaled here -- without it the
    # caller's 160-370 K valid_mask rejects every real pixel.
    raw = xr.DataArray(np.array([[16000.0, 29000.0], [37000.0, np.nan]]), dims=("y", "x"))
    raw.attrs["long_name"] = "Ta_mean"

    scaled = glass_modis_module._scale_bands({"Ta_mean": raw}, 0.01)["Ta_mean"]
    assert float(scaled.isel(y=0, x=0)) == 160.0
    assert float(scaled.isel(y=1, x=0)) == 370.0
    assert "scale_factor" not in scaled.attrs  # stale attr dropped, no double-apply

    # a scale_factor on the band's own attrs wins over the passed default
    raw2 = raw.copy()
    raw2.attrs["scale_factor"] = 0.1
    assert float(glass_modis_module._scale_bands({"b": raw2}, 0.01)["b"].isel(y=0, x=0)) == 1600.0

    # no-op passthrough
    assert glass_modis_module._scale_bands({"b": raw}, 1.0)["b"] is raw
    assert glass_modis_module._scale_bands({}, 0.01) == {}


def test_open_hdf_bands_scales_raw_dn_to_physical(monkeypatch):
    # End-to-end through _open_hdf_bands: a fake reader returns raw DN, the
    # returned bands must be in physical Kelvin.
    fake = _fake_multiband([3700.0, 2900.0, 1600.0], 3, long_name=("Ta_max", "Ta_mean", "Ta_min"))
    monkeypatch.setattr(glass_modis_module.rxr, "open_rasterio", lambda path, masked=True: fake)

    bands = glass_modis_module._open_hdf_bands("fake.hdf", ("Ta_min", "Ta_mean", "Ta_max"), scale_factor=0.01)

    assert float(bands["Ta_max"].isel(y=0, x=0)) == 37.0
    assert float(bands["Ta_mean"].isel(y=0, x=0)) == 29.0
    assert float(bands["Ta_min"].isel(y=0, x=0)) == 16.0


def test_open_hdf_bands_matches_grouped_multiband_array_by_long_name(monkeypatch):
    # The real production shape for Ta (2026-08-17 bug): GDAL groups
    # Ta_min/Ta_mean/Ta_max into one 3-band DataArray, not a list. GDAL's
    # own per-band long_name metadata (when present) should be used to
    # identify each band correctly regardless of physical order.
    fake = _fake_multiband([300.0, 290.0, 280.0], 3, long_name=("Ta_max", "Ta_mean", "Ta_min"))
    monkeypatch.setattr(glass_modis_module.rxr, "open_rasterio", lambda path, masked=True: fake)

    bands = glass_modis_module._open_hdf_bands("fake.hdf", ("Ta_min", "Ta_mean", "Ta_max"))

    assert set(bands) == {"Ta_min", "Ta_mean", "Ta_max"}
    assert float(bands["Ta_max"].isel(y=0, x=0)) == 300.0
    assert float(bands["Ta_min"].isel(y=0, x=0)) == 280.0


def test_open_hdf_bands_falls_back_to_physical_order_without_long_name(monkeypatch):
    # No usable long_name/name metadata at all -- must fall back to the
    # confirmed real physical order (Max, Mean, Min) rather than returning
    # {} (the pre-fix bug: an empty dict silently dropped every day).
    fake = _fake_multiband([300.0, 290.0, 280.0], 3)
    monkeypatch.setattr(glass_modis_module.rxr, "open_rasterio", lambda path, masked=True: fake)

    bands = glass_modis_module._open_hdf_bands("fake.hdf", ("Ta_min", "Ta_mean", "Ta_max"))

    assert set(bands) == {"Ta_min", "Ta_mean", "Ta_max"}
    assert float(bands["Ta_max"].isel(y=0, x=0)) == 300.0
    assert float(bands["Ta_min"].isel(y=0, x=0)) == 280.0


def test_execute_fetch_writes_annual_geotiff_with_eight_bands_lst(tmp_path, monkeypatch):
    source, _ = _make_source(tmp_path, day_range={"start": [2019, 1], "end": [2019, 3]})
    target = source.plan(PipelineStep.FETCH, TargetSelection(keys=("2019/h08v05",)))[0]

    def fake_listing_for(year, day):
        href = f"GLASS06A01.V01.A{year}{day:03d}.h08v05.2022021.hdf"
        return [(href, f"https://x/{href}")]

    monkeypatch.setattr(source, "_listing_for", fake_listing_for)
    monkeypatch.setattr(source, "download_async", _fake_download_async)
    monkeypatch.setattr(
        glass_modis_module, "_open_hdf_bands", lambda path, band_names, scale_factor=1.0: {"LST": _fake_band(290.0)}
    )

    assert source.execute(target) is True
    assert os.path.exists(target.output_path)

    da = rxr.open_rasterio(target.output_path)
    assert da.sizes["band"] == 8
    assert da.rio.crs is not None


def test_execute_fetch_writes_annual_geotiff_with_eight_bands_ta(tmp_path, monkeypatch):
    source, _ = _make_source(tmp_path, "glass_ta_modis", day_range={"start": [2019, 1], "end": [2019, 3]})
    target = source.plan(PipelineStep.FETCH, TargetSelection(keys=("2019/h08v05",)))[0]

    def fake_listing_for(year, day):
        href = f"GLASS18A01.V10.A{year}{day:03d}.h08v05.2023300.hdf"
        return [(href, f"https://x/{href}")]

    monkeypatch.setattr(source, "_listing_for", fake_listing_for)
    monkeypatch.setattr(source, "download_async", _fake_download_async)
    monkeypatch.setattr(
        glass_modis_module,
        "_open_hdf_bands",
        lambda path, band_names, scale_factor=1.0: {
            "Ta_min": _fake_band(280.0), "Ta_mean": _fake_band(290.0), "Ta_max": _fake_band(300.0)
        },
    )

    assert source.execute(target) is True
    da = rxr.open_rasterio(target.output_path)
    assert da.sizes["band"] == 8


def test_execute_fetch_treats_missing_day_as_a_gap_not_a_failure(tmp_path, monkeypatch):
    # A day genuinely absent from the remote listing (sensor gap) should not
    # fail the whole tile-year target -- only the days that ARE present feed
    # the annual stats.
    source, _ = _make_source(tmp_path, day_range={"start": [2019, 1], "end": [2019, 3]})
    target = source.plan(PipelineStep.FETCH, TargetSelection(keys=("2019/h08v05",)))[0]

    def fake_listing_for(year, day):
        if day == 2:
            return []  # absent -- a gap
        href = f"GLASS06A01.V01.A{year}{day:03d}.h08v05.2022021.hdf"
        return [(href, f"https://x/{href}")]

    monkeypatch.setattr(source, "_listing_for", fake_listing_for)
    monkeypatch.setattr(source, "download_async", _fake_download_async)
    monkeypatch.setattr(
        glass_modis_module, "_open_hdf_bands", lambda path, band_names, scale_factor=1.0: {"LST": _fake_band(290.0)}
    )

    assert source.execute(target) is True
    assert os.path.exists(target.output_path)


def test_execute_fetch_fails_when_no_days_found(tmp_path, monkeypatch):
    source, _ = _make_source(tmp_path, day_range={"start": [2019, 1], "end": [2019, 3]})
    target = source.plan(PipelineStep.FETCH, TargetSelection(keys=("2019/h08v05",)))[0]

    monkeypatch.setattr(source, "_listing_for", lambda year, day: [])

    assert source.execute(target) is False
    assert not os.path.exists(target.output_path)


def test_fetch_concurrency_knob(tmp_path):
    # one knob drives the semaphore + limit_per_host; falls back to
    # max_concurrent_downloads, then to the default of 4.
    assert _make_source(tmp_path)[0].fetch_concurrency == 4
    assert _make_source(tmp_path, fetch_concurrency=8)[0].fetch_concurrency == 8
    s = _make_source(tmp_path, max_concurrent_downloads=3)[0]
    assert s.fetch_concurrency == 3 and s.max_concurrent_downloads == 3


def test_prefetch_listings_warms_cache_concurrently(tmp_path):
    import asyncio

    source, _ = _make_source(tmp_path, day_range={"start": [2019, 1], "end": [2019, 5]})

    calls = []

    def fake_list_single_directory(url):
        calls.append(url)
        return [("GLASS06A01.V01.A2019001.h08v05.x.hdf", f"{url}f.hdf")]

    # patch the underlying GET; real _listing_for still runs (+ caches)
    source._list_single_directory = fake_list_single_directory

    asyncio.run(source._prefetch_listings(2019))
    assert len(calls) == 5                      # one GET per day, once
    assert set(source._listing_cache) == {(2019, d) for d in range(1, 6)}

    calls.clear()
    asyncio.run(source._prefetch_listings(2019))  # all cached now
    assert calls == []


def test_prefetch_listings_caches_404_day_as_empty(tmp_path):
    import asyncio
    import requests

    source, _ = _make_source(tmp_path, day_range={"start": [2019, 1], "end": [2019, 2]})

    def boom(url):
        resp = requests.Response()
        resp.status_code = 404
        raise requests.HTTPError(response=resp)

    source._list_single_directory = boom
    asyncio.run(source._prefetch_listings(2019))
    assert source._listing_cache == {(2019, 1): [], (2019, 2): []}
