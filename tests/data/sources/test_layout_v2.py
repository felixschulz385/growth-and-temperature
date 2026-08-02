"""Tests for `grid_store_path()`, the `layout: v2` single-source-family
rename (docs/design/09-integrated-pipeline.md §14's deferred task,
docs/design/02-storage.md §2's "one store per variable family" decision).

Kept separate from tests/data/sources/test_layout.py, which pins the legacy
`output_root()`/`raw_root()`/`index_path()` byte-for-byte -- this file only
exercises the new, additive `layout=v2` behaviour.
"""

import os

from src.data.sources.layout import grid_store_path
from src.data.sources.steps import PipelineStep


def test_default_layout_matches_legacy_output_root_plus_filename():
    assert grid_store_path("/data", "acag/pm25", "acag_pm25_timeseries_reprojected.zarr") == os.path.join(
        "/data", "acag/pm25", "processed", "stage_2", "acag_pm25_timeseries_reprojected.zarr"
    )


def test_v2_layout_with_family_uses_shared_grid_v2_directory():
    assert grid_store_path(
        "/data", "acag/pm25", "acag_pm25_timeseries_reprojected.zarr", layout="v2", v2_family="pm25"
    ) == os.path.join("/data", "grid_v2", "pm25.zarr")


def test_v2_layout_without_family_falls_back_to_legacy_path():
    # A source not yet part of the single-source-family scope (e.g. one of
    # the deferred multi-source families) must not silently break just
    # because layout=v2 was selected globally.
    assert grid_store_path(
        "/data", "eog/viirs", "eog_viirs_timeseries_reprojected.zarr", layout="v2", v2_family=None
    ) == os.path.join("/data", "eog/viirs", "processed", "stage_2", "eog_viirs_timeseries_reprojected.zarr")


def test_v2_layout_respects_grid_id_when_falling_back():
    assert grid_store_path(
        "/data", "eog/viirs", "eog_viirs_timeseries_reprojected.zarr", layout="v2", grid_id="ease6933"
    ) == os.path.join("/data", "eog/viirs", "processed", "stage_2_ease6933", "eog_viirs_timeseries_reprojected.zarr")


def test_v2_family_ignored_under_legacy_layout():
    # v2_family is only consulted when layout=v2 -- passing it under the
    # default legacy layout must not change anything.
    assert grid_store_path(
        "/data", "acag/pm25", "acag_pm25_timeseries_reprojected.zarr", v2_family="pm25"
    ) == os.path.join("/data", "acag/pm25", "processed", "stage_2", "acag_pm25_timeseries_reprojected.zarr")
