"""Tests for `grid_store_path()`, the `layout: v2` single-source-family
rename (docs/design/09-integrated-pipeline.md §14's deferred task,
docs/design/02-storage.md §2's "one store per variable family" decision).

Kept separate from tests/data/sources/test_layout.py, which pins the legacy
`output_root()`/`raw_root()`/`index_path()` byte-for-byte -- this file only
exercises the new, additive `layout=v2` behaviour.
"""

import os

from src.data.sources.layout import grid_store_path, output_root, raw_root
from src.data.sources.steps import PipelineStep


def test_default_layout_matches_legacy_output_root_plus_filename():
    assert grid_store_path("/data", "acag/pm25", "acag_pm25_timeseries_reprojected.zarr") == os.path.join(
        "/data", "acag/pm25", "processed", "stage_2", "acag_pm25_timeseries_reprojected.zarr"
    )


def test_v2_layout_with_family_uses_grid_slash_grid_id_directory():
    assert grid_store_path(
        "/data", "acag/pm25", "acag_pm25_timeseries_reprojected.zarr", layout="v2", v2_family="pm25"
    ) == os.path.join("/data", "grid", "legacy_4326", "pm25.zarr")


def test_v2_layout_with_family_folds_grid_id_into_the_path():
    assert grid_store_path(
        "/data", "acag/pm25", "acag_pm25_timeseries_reprojected.zarr",
        layout="v2", grid_id="ease6933", v2_family="pm25",
    ) == os.path.join("/data", "grid", "ease6933", "pm25.zarr")


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


def test_all_in_use_v2_family_names_are_unique():
    # Every literal (or f-string-resolved) v2_family= value passed to
    # grid_store_path() across all sources, kept in sync by hand -- cheap
    # insurance against a future copy-paste collision in the grid/<grid_id>/
    # namespace. If this test fails, two different sources would silently
    # overwrite each other's store under layout=v2.
    in_use_families = [
        "pm25",  # acag.py
        "land_cover",  # esacci.py
        "land_mask",  # misc/osm.py
        "country_id",  # misc/gadm.py (misc/country_classifications.py only reads this one)
        "classifications",  # misc/country_classifications.py
        "admin_panel_adm1",  # plad.py, admin_level=1
        "admin_panel_adm2",  # plad.py, admin_level=2
        "eog_dmsp",  # eog/source.py, source_type="dmsp"
        "eog_viirs_annual",  # eog/source.py, source_type="viirs_annual"
        "eog_viirs_dvnl",  # eog/source.py, source_type="viirs_dvnl"
        "ntl_harm",  # ntl_harm.py
        "modis_lst_21a2",  # modis/source.py, product="21A2"
        "modis_lst_11a1",  # modis/source.py, product="11A1"
        "glass_modis_lst",  # glass/source.py, MODIS variant
        "glass_avhrr_lst",  # glass/source.py, AVHRR variant
        "berman_mining",  # berman_mining.py
        "snl_mining",  # snl_mining/source.py
    ]
    assert len(in_use_families) == len(set(in_use_families))


def test_v2_family_ignored_under_legacy_layout():
    # v2_family is only consulted when layout=v2 -- passing it under the
    # default legacy layout must not change anything.
    assert grid_store_path(
        "/data", "acag/pm25", "acag_pm25_timeseries_reprojected.zarr", v2_family="pm25"
    ) == os.path.join("/data", "acag/pm25", "processed", "stage_2", "acag_pm25_timeseries_reprojected.zarr")


def test_raw_root_v2_layout_flips_to_top_level_raw_tree():
    assert raw_root("/data", "acag/pm25", layout="v2") == os.path.join("/data", "raw", "acag/pm25")


def test_raw_root_v2_layout_applies_namespace():
    assert raw_root("/data", "misc", namespace="gadm", layout="v2") == os.path.join("/data", "raw", "misc", "gadm")


def test_raw_root_legacy_layout_unchanged():
    assert raw_root("/data", "acag/pm25", layout="legacy") == os.path.join("/data", "acag/pm25", "raw")


def test_output_root_prepare_v2_layout_flips_to_top_level_prepared_tree():
    assert output_root("/data", "acag/pm25", PipelineStep.PREPARE, layout="v2") == os.path.join(
        "/data", "prepared", "acag/pm25"
    )


def test_output_root_prepare_v2_layout_applies_namespace():
    assert output_root(
        "/data", "misc", PipelineStep.PREPARE, namespace="osm", layout="v2"
    ) == os.path.join("/data", "prepared", "misc", "osm")


def test_output_root_prepare_legacy_layout_unchanged():
    assert output_root("/data", "acag/pm25", PipelineStep.PREPARE, layout="legacy") == os.path.join(
        "/data", "acag/pm25", "processed", "stage_1"
    )


def test_output_root_grid_v2_layout_ignores_namespace():
    # GRID's v2 directory is flat (grid/<grid_id>) -- namespace is only
    # meaningful for the legacy per-source layout.
    assert output_root(
        "/data", "misc", PipelineStep.GRID, namespace="gadm", layout="v2", grid_id="ease6933"
    ) == os.path.join("/data", "grid", "ease6933")


def test_output_root_fetch_v2_layout_matches_raw_root():
    assert output_root("/data", "eog/viirs", PipelineStep.FETCH, layout="v2") == raw_root(
        "/data", "eog/viirs", layout="v2"
    )
