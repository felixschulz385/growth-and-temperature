"""Characterization tests for the *current* MiscPreprocessor (pre-migration).

docs/design/09-integrated-pipeline.md §10 step 0 / §7 (the misc split).
Covers target generation only -- the heavy rasterization methods
(_rasterize_osm_target, _rasterize_gadm_target's tiled processing,
_process_country_classifications_target's HDI/WB parsing) are ported
mechanically in the split sources without new fast-unit-test surface here,
consistent with how this migration has treated every source's heavy compute.
"""

import os

import pandas as pd

from src.data.preprocess.sources.misc import MiscPreprocessor

_SOURCES_CONFIG = {
    "misc": {
        "type": "misc",
        "data_path": "misc",
        "sources": {
            "osm": {"url": "https://osmdata.example/land-polygons.zip", "name": "land-polygons.zip", "subfolder": "osm"},
            "gadm": {"url": "https://geodata.example/gadm.zip", "name": "gadm.zip", "subfolder": "gadm"},
            "hdi": {"url": "https://hdr.example/HDR25.csv", "name": "HDR25.csv", "subfolder": "hdi"},
            "worldbank_income_classes": {"url": "https://ddh.example/DR0095334.xlsx", "name": "DR0095334.xlsx", "subfolder": "hdi"},
        },
    }
}


def _write_index(index_dir, data_path, rows):
    safe = data_path.replace("/", "_").replace("\\", "_")
    os.makedirs(index_dir, exist_ok=True)
    pd.DataFrame(rows).to_parquet(os.path.join(index_dir, f"parquet_{safe}.parquet"))


def _make_preprocessor(tmp_path, subsource=None, rows=None):
    hpc_root = str(tmp_path)
    if rows is None:
        rows = [
            {"relative_path": "land-polygons.zip", "status_category": "completed"},
            {"relative_path": "gadm.zip", "status_category": "completed"},
            {"relative_path": "HDR25.csv", "status_category": "completed"},
            {"relative_path": "DR0095334.xlsx", "status_category": "completed"},
        ]
    _write_index(os.path.join(hpc_root, "hpc_data_index"), "misc", rows)
    kwargs = dict(hpc_target=hpc_root, name="misc", sources=_SOURCES_CONFIG)
    if subsource:
        kwargs["subsource"] = subsource
    return MiscPreprocessor(**kwargs)


def test_hpc_output_path(tmp_path):
    pp = _make_preprocessor(tmp_path)
    assert pp.get_hpc_output_path("vector") == os.path.join(str(tmp_path), "misc", "processed", "stage_1")
    assert pp.get_hpc_output_path("spatial") == os.path.join(str(tmp_path), "misc", "processed", "stage_2")


def test_vector_targets_categorized_by_filename_pattern(tmp_path):
    pp = _make_preprocessor(tmp_path)
    targets = pp.get_preprocessing_targets("vector")
    by_type = {t["data_type"]: t for t in targets}
    assert set(by_type) == {"osm", "gadm", "country_classifications"}
    assert by_type["osm"]["output_path"] == f"{pp.get_hpc_output_path('vector')}/osm/land_polygons_simplified.gpkg"
    assert by_type["gadm"]["output_path"] == f"{pp.get_hpc_output_path('vector')}/gadm/gadm_levels_simplified.gpkg"
    assert by_type["country_classifications"]["output_path"] == (
        f"{pp.get_hpc_output_path('vector')}/country_classifications/classifications.parquet"
    )
    assert by_type["country_classifications"]["metadata"]["has_hdi"] is True
    assert by_type["country_classifications"]["metadata"]["has_wb"] is True
    assert sorted(os.path.basename(f) for f in by_type["country_classifications"]["source_files"]) == [
        "DR0095334.xlsx", "HDR25.csv",
    ]


def test_subsource_filter_restricts_to_one_group(tmp_path):
    pp = _make_preprocessor(tmp_path, subsource="osm")
    targets = pp.get_preprocessing_targets("vector")
    assert {t["data_type"] for t in targets} == {"osm"}


def test_subsource_filter_resolves_individual_key_to_group(tmp_path):
    # "hdi" (an individual source key) resolves to the "country_classifications" group.
    pp = _make_preprocessor(tmp_path, subsource="hdi")
    targets = pp.get_preprocessing_targets("vector")
    assert {t["data_type"] for t in targets} == {"country_classifications"}


def test_spatial_targets_depend_on_vector_outputs_existing_on_disk(tmp_path):
    pp = _make_preprocessor(tmp_path, subsource="osm")
    # No vector output on disk yet -> no spatial target.
    assert pp.get_preprocessing_targets("spatial") == []

    vector_dir = os.path.join(pp.get_hpc_output_path("vector"), "osm")
    os.makedirs(vector_dir, exist_ok=True)
    open(os.path.join(vector_dir, "land_polygons_simplified.gpkg"), "w").close()

    targets = pp.get_preprocessing_targets("spatial")
    assert len(targets) == 1
    assert targets[0]["output_path"] == f"{pp.get_hpc_output_path('spatial')}/osm/land_mask.zarr"


def test_country_classifications_spatial_target_requires_gadm_grid(tmp_path):
    pp = _make_preprocessor(tmp_path, subsource="country_classifications")
    vector_dir = os.path.join(pp.get_hpc_output_path("vector"), "country_classifications")
    os.makedirs(vector_dir, exist_ok=True)
    open(os.path.join(vector_dir, "classifications.parquet"), "w").close()

    # GADM grid not yet present -> no target.
    assert pp.get_preprocessing_targets("spatial") == []

    gadm_dir = os.path.join(pp.get_hpc_output_path("spatial"), "gadm")
    os.makedirs(gadm_dir, exist_ok=True)
    open(os.path.join(gadm_dir, "countries_grid.zarr"), "w").close()

    targets = pp.get_preprocessing_targets("spatial")
    assert len(targets) == 1
    assert targets[0]["output_path"] == f"{pp.get_hpc_output_path('spatial')}/country_classifications/classifications_grid.zarr"
