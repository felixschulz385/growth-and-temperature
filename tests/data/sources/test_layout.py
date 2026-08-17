"""Pin every path layout.output_root()/raw_root()/grid_store_path()/index_path()
can produce -- the stage-name-first physical tree (`raw/<data_path>`,
`prepared/<data_path>`, `grid/<grid_id>/<family>.zarr`; see
`src/data/sources/layout.py`'s module docstring).
"""

import os

from src.data.sources.layout import grid_store_path, index_path, output_root, raw_root
from src.data.sources.steps import PipelineStep


def test_raw_root_shape():
    assert raw_root("/data", "acag/pm25") == os.path.join("/data", "raw", "acag/pm25")


def test_raw_root_applies_namespace():
    assert raw_root("/data", "misc", namespace="gadm") == os.path.join("/data", "raw", "misc", "gadm")


def test_output_root_prepare_shape():
    assert output_root("/data", "acag/pm25", PipelineStep.PREPARE) == os.path.join(
        "/data", "prepared", "acag/pm25"
    )


def test_output_root_prepare_applies_namespace():
    assert output_root("/data", "misc", PipelineStep.PREPARE, namespace="osm") == os.path.join(
        "/data", "prepared", "misc", "osm"
    )


def test_output_root_grid_shape():
    assert output_root("/data", "acag/pm25", PipelineStep.GRID) == os.path.join(
        "/data", "grid", "legacy_4326"
    )


def test_output_root_grid_uses_requested_grid_id():
    assert output_root("/data", "modis", PipelineStep.GRID, grid_id="ease6933") == os.path.join(
        "/data", "grid", "ease6933"
    )


def test_output_root_grid_ignores_namespace():
    # GRID's directory is flat (grid/<grid_id>) -- namespace is only
    # meaningful for FETCH/PREPARE.
    assert output_root(
        "/data", "misc", PipelineStep.GRID, namespace="gadm", grid_id="ease6933"
    ) == os.path.join("/data", "grid", "ease6933")


def test_output_root_fetch_matches_raw_root():
    assert output_root("/data", "eog/viirs", PipelineStep.FETCH) == raw_root("/data", "eog/viirs")


def test_grid_store_path_uses_grid_slash_grid_id_directory():
    assert grid_store_path("/data", "acag/pm25", family="pm25") == os.path.join(
        "/data", "grid", "legacy_4326", "pm25.zarr"
    )


def test_grid_store_path_folds_grid_id_into_the_path():
    assert grid_store_path("/data", "acag/pm25", grid_id="ease6933", family="pm25") == os.path.join(
        "/data", "grid", "ease6933", "pm25.zarr"
    )


def test_all_in_use_family_names_are_unique():
    # Every literal (or f-string-resolved) family= value passed to
    # grid_store_path() across all sources, kept in sync by hand -- cheap
    # insurance against a future copy-paste collision in the grid/<grid_id>/
    # namespace. If this test fails, two different sources would silently
    # overwrite each other's store.
    in_use_families = [
        "pm25",  # acag.py
        "land_cover",  # esacci.py
        "land_mask",  # misc/osm.py
        "country_id",  # misc/gadm.py (misc/country_classifications.py only reads this one)
        "ecoregions",  # ecoregions/source.py
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
