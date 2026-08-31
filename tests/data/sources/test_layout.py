"""Pin every path layout.output_root()/raw_root()/grid_store_path()/index_path()
can produce -- the stage-name-first physical tree (`raw/<data_path>`,
`prepared/<data_path>/<agg>/...`, pixel-grid stores under
`prepared/<data_path>/crs/<grid_id>/<family>.zarr`; see
`src/data/sources/layout.py`'s module docstring).
"""

import os

import pytest

from src.data.sources.layout import ADM_AGG, CRS_AGG, MISC_AGG, grid_store_path, index_path, output_root, raw_root
from src.data.sources.steps import PipelineStep


def test_raw_root_shape():
    assert raw_root("/data", "acag/pm25") == os.path.join("/data", "raw", "acag/pm25")


def test_raw_root_applies_namespace():
    assert raw_root("/data", "misc", namespace="gadm") == os.path.join("/data", "raw", "misc", "gadm")


def test_output_root_prepare_shape():
    assert output_root("/data", "acag/pm25", PipelineStep.PREPARE, agg=CRS_AGG) == os.path.join(
        "/data", "prepared", "acag/pm25", "crs"
    )


def test_output_root_prepare_applies_namespace():
    assert output_root("/data", "misc", PipelineStep.PREPARE, namespace="osm", agg=MISC_AGG) == os.path.join(
        "/data", "prepared", "misc", "misc", "osm"
    )


def test_output_root_prepare_requires_agg():
    with pytest.raises(ValueError):
        output_root("/data", "acag/pm25", PipelineStep.PREPARE)


def test_output_root_prepare_supports_all_three_agg_buckets():
    assert output_root("/data", "x", PipelineStep.PREPARE, agg=CRS_AGG).endswith(os.path.join("x", "crs"))
    assert output_root("/data", "x", PipelineStep.PREPARE, agg=ADM_AGG).endswith(os.path.join("x", "adm"))
    assert output_root("/data", "x", PipelineStep.PREPARE, agg=MISC_AGG).endswith(os.path.join("x", "misc"))


def test_output_root_grid_shape():
    assert output_root("/data", "acag/pm25", PipelineStep.GRID) == os.path.join(
        "/data", "prepared", "acag/pm25", "crs", "legacy_4326"
    )


def test_output_root_grid_uses_requested_grid_id():
    assert output_root("/data", "modis", PipelineStep.GRID, grid_id="ease6933") == os.path.join(
        "/data", "prepared", "modis", "crs", "ease6933"
    )


def test_output_root_grid_ignores_namespace():
    # GRID's directory is flat (prepared/<data_path>/crs/<grid_id>) --
    # namespace is only meaningful for FETCH/PREPARE.
    assert output_root(
        "/data", "misc", PipelineStep.GRID, namespace="gadm", grid_id="ease6933"
    ) == os.path.join("/data", "prepared", "misc", "crs", "ease6933")


def test_output_root_fetch_matches_raw_root():
    assert output_root("/data", "eog/viirs", PipelineStep.FETCH) == raw_root("/data", "eog/viirs")


def test_grid_store_path_uses_prepared_crs_grid_id_directory():
    assert grid_store_path("/data", "acag/pm25", family="pm25") == os.path.join(
        "/data", "prepared", "acag/pm25", "crs", "legacy_4326", "pm25.zarr"
    )


def test_grid_store_path_folds_grid_id_into_the_path():
    assert grid_store_path("/data", "acag/pm25", grid_id="ease6933", family="pm25") == os.path.join(
        "/data", "prepared", "acag/pm25", "crs", "ease6933", "pm25.zarr"
    )


def test_all_in_use_family_names_are_unique():
    # Every literal (or f-string-resolved) family= value passed to
    # grid_store_path() across all sources, kept in sync by hand -- cheap
    # insurance against a future copy-paste collision in the
    # prepared/<data_path>/crs/<grid_id>/ namespace. If this test fails, two
    # different sources would silently overwrite each other's store.
    in_use_families = [
        "pm25",  # acag.py
        "land_cover",  # esacci.py
        "land_mask",  # misc/osm.py
        "country_id",  # misc/gadm.py (misc/country_classifications.py only reads this one)
        "ecoregions",  # ecoregions/source.py
        # plad.py's PREPARE output is a per-instance-filename ADM_AGG table
        # (plad_adm{admin_level}_reg_fav.parquet), not a grid_store_path()
        # family -- no entry needed here.
        "eog_dmsp",  # eog/source.py, source_type="dmsp"
        "eog_viirs_annual",  # eog/source.py, source_type="viirs_annual"
        "eog_viirs_dvnl",  # eog/source.py, source_type="viirs_dvnl"
        "eog_flare",  # eog/flare.py
        "ntl_harm",  # ntl_harm.py
        "modis_lst_21a2",  # modis/source.py, product="21A2"
        "modis_lst_11a1",  # modis/source.py, product="11A1"
        "glass_modis_lst",  # glass/modis.py, "lst" variant
        "glass_modis_ta",  # glass/modis.py, "ta" variant
        "glass_avhrr_lst",  # glass/avhrr.py
        "berman_mining",  # berman_mining.py
        "snl_mining",  # snl_mining/source.py
    ]
    assert len(in_use_families) == len(set(in_use_families))


def test_index_path_mirrors_unified_data_index_filename_derivation():
    # UnifiedDataIndex derives parquet_<safe(data_path)>.parquet from
    # data_source.data_path -- unchanged by this refactor, so acag's index
    # stays byte-identical (data_path="acag/pm25").
    assert index_path("/idx", "acag/pm25") == "/idx/parquet_acag_pm25.parquet"
    # The misc split gives osm/gadm/country_classifications distinct
    # data_path values so each gets its own index file, unlike today's single
    # shared parquet_misc.parquet.
    assert index_path("/idx", "misc/gadm") == "/idx/parquet_misc_gadm.parquet"


def test_index_path_returns_none_when_local_index_dir_unset():
    # paths.local_index_dir left unset in data.yaml (empty string/omitted)
    # resolves to None on PipelineContext -- callers (every source's
    # _plan_prepare()) must get None back, not a TypeError from
    # os.path.join(None, ...), so they can treat "no index configured" the
    # same as "index file not found" (warn + return no targets).
    assert index_path(None, "acag/pm25") is None
    assert index_path("", "acag/pm25") is None
