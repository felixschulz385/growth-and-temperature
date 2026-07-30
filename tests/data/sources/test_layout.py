"""Pin every path layout.output_root()/raw_root()/index_path() can produce.

docs/design/09-integrated-pipeline.md §3: this is the anti-regression net for
the "do not rename physical directories" decision. Every value here must match
today's src/data/preprocess/sources/*::get_hpc_output_path() /
src/data/preprocess/sources/*::_resolve_raw_path() output exactly -- see
tests/data/preprocess/sources/test_characterization_acag.py for the values
pinned directly against the old code.
"""

import os

from src.data.sources.layout import index_path, output_root, raw_root
from src.data.sources.steps import PipelineStep


def test_fetch_root_matches_acag_resolve_raw_path():
    # src/data/preprocess/sources/acag.py:163-167
    assert raw_root("/data", "acag/pm25") == os.path.join("/data", "acag/pm25", "raw")


def test_prepare_root_matches_acag_stage_1():
    # src/data/preprocess/sources/acag.py:186-193, stage="annual"
    assert output_root("/data", "acag/pm25", PipelineStep.PREPARE) == os.path.join(
        "/data", "acag/pm25", "processed", "stage_1"
    )


def test_grid_root_matches_acag_stage_2_legacy_grid():
    # src/data/preprocess/sources/acag.py:186-193, stage="spatial"
    assert output_root("/data", "acag/pm25", PipelineStep.GRID) == os.path.join(
        "/data", "acag/pm25", "processed", "stage_2"
    )


def test_grid_root_uses_ease6933_suffix_when_requested():
    # docs/design/05-migration.md §1's additive stage_2_ease6933 path,
    # today only honoured ad hoc by MODIS.
    assert output_root(
        "/data", "modis", PipelineStep.GRID, grid_id="ease6933"
    ) == os.path.join("/data", "modis", "processed", "stage_2_ease6933")


def test_fetch_root_equals_output_root_for_fetch_step():
    assert output_root("/data", "eog/viirs", PipelineStep.FETCH) == raw_root("/data", "eog/viirs")


def test_namespace_inserted_after_stage_dir_for_misc_split_sources():
    # misc's gadm/osm subsources rasterize under
    # misc/processed/stage_2/<namespace>/... today -- the split keeps the
    # same physical layout via namespace= (docs/design/09-integrated-pipeline.md §7).
    assert output_root("/data", "misc", PipelineStep.GRID, namespace="gadm") == os.path.join(
        "/data", "misc", "processed", "stage_2", "gadm"
    )
    assert output_root("/data", "misc", PipelineStep.PREPARE, namespace="osm") == os.path.join(
        "/data", "misc", "processed", "stage_1", "osm"
    )


def test_index_path_mirrors_unified_data_index_filename_derivation():
    # UnifiedDataIndex derives parquet_<safe(data_path)>.parquet from
    # data_source.data_path -- unchanged by this refactor, so acag's index
    # stays byte-identical (data_path="acag/pm25").
    assert index_path("/idx", "acag/pm25") == "/idx/parquet_acag_pm25.parquet"
    # The misc split gives osm/gadm/country_classifications distinct
    # data_path values so each gets its own index file, unlike today's single
    # shared parquet_misc.parquet.
    assert index_path("/idx", "misc/gadm") == "/idx/parquet_misc_gadm.parquet"
