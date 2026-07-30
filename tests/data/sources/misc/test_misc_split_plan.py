"""OsmSource/GadmSource/CountryClassificationsSource.plan() must reproduce
the relevant slice of the old MiscPreprocessor's targets.
Oracle: tests/data/preprocess/sources/test_characterization_misc.py.
"""

import os

from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources import registry
from src.data.sources.steps import MissingPrerequisiteError, PipelineStep, TargetSelection


def _ctx(tmp_path):
    return PipelineContext(data_root=str(tmp_path / "data_root"), local_index_dir=str(tmp_path / "index"))


def _make(tmp_path, source_id, **raw):
    ctx = _ctx(tmp_path)
    cfg = SourceConfig.from_dict(source_id, dict(raw))
    cls = registry.load(source_id)
    return cls(ctx, cfg), ctx


def test_osm_and_gadm_share_output_root_but_have_distinct_index_data_path(tmp_path):
    osm, ctx = _make(tmp_path, "osm")
    gadm, _ = _make(tmp_path, "gadm")

    assert osm.output_root(PipelineStep.PREPARE) == os.path.join(ctx.data_root, "misc", "processed", "stage_1", "osm")
    assert gadm.output_root(PipelineStep.PREPARE) == os.path.join(ctx.data_root, "misc", "processed", "stage_1", "gadm")
    # Distinct index files -- the actual point of the split.
    assert osm.data_path == "misc/osm"
    assert gadm.data_path == "misc/gadm"
    assert osm.data_path != gadm.data_path


def test_osm_prepare_target(tmp_path):
    osm, ctx = _make(tmp_path, "osm")
    raw_dir = osm.output_root(PipelineStep.FETCH)
    os.makedirs(raw_dir, exist_ok=True)
    open(os.path.join(raw_dir, osm.CONFIGURED_FILES[0].name), "w").close()

    targets = osm.plan(PipelineStep.PREPARE, TargetSelection())
    assert len(targets) == 1
    assert targets[0].output_path == os.path.join(
        ctx.data_root, "misc", "processed", "stage_1", "osm", "land_polygons_simplified.gpkg"
    )


def test_osm_grid_target_depends_on_prepare_output(tmp_path):
    osm, _ = _make(tmp_path, "osm")
    assert osm.plan(PipelineStep.GRID, TargetSelection()) == []

    vector_dir = osm.output_root(PipelineStep.PREPARE)
    os.makedirs(vector_dir, exist_ok=True)
    open(os.path.join(vector_dir, "land_polygons_simplified.gpkg"), "w").close()

    targets = osm.plan(PipelineStep.GRID, TargetSelection())
    assert len(targets) == 1
    assert targets[0].output_path == os.path.join(osm.output_root(PipelineStep.GRID), "land_mask.zarr")


def test_gadm_grid_target_includes_adm1_when_present(tmp_path):
    gadm, _ = _make(tmp_path, "gadm")
    vector_dir = gadm.output_root(PipelineStep.PREPARE)
    os.makedirs(vector_dir, exist_ok=True)
    open(os.path.join(vector_dir, "gadm_levelADM_0_simplified.gpkg"), "w").close()

    targets = gadm.plan(PipelineStep.GRID, TargetSelection())
    assert len(targets) == 1
    assert len(targets[0].inputs) == 1  # no ADM_1 file yet

    open(os.path.join(vector_dir, "gadm_levelADM_1_simplified.gpkg"), "w").close()
    targets = gadm.plan(PipelineStep.GRID, TargetSelection())
    assert len(targets[0].inputs) == 2


def test_country_classifications_prepare_target_tracks_which_sources_present(tmp_path):
    cc, _ = _make(tmp_path, "country_classifications")
    raw_dir = cc.output_root(PipelineStep.FETCH)
    os.makedirs(raw_dir, exist_ok=True)
    open(os.path.join(raw_dir, cc._raw_file("hdi").rsplit("/", 1)[-1]), "w").close()
    # Only HDI present.
    open(cc._raw_file("hdi"), "w").close()

    targets = cc.plan(PipelineStep.PREPARE, TargetSelection())
    assert len(targets) == 1
    assert targets[0].meta == {"has_hdi": True, "has_wb": False}


def test_country_classifications_grid_requires_gadm_output_via_shared_layout(tmp_path):
    cc, ctx = _make(tmp_path, "country_classifications")
    vector_dir = cc.output_root(PipelineStep.PREPARE)
    os.makedirs(vector_dir, exist_ok=True)
    open(os.path.join(vector_dir, "classifications.parquet"), "w").close()

    # GADM's grid output not yet present -> no target (this is the plan()-level
    # check; the runner's REQUIRES enforcement, via MissingPrerequisiteError,
    # is a separate, earlier gate -- see src/cli/pipeline/handlers.py::_check_requires).
    assert cc.plan(PipelineStep.GRID, TargetSelection()) == []

    gadm_grid_dir = os.path.join(ctx.data_root, "misc", "processed", "stage_2", "gadm")
    os.makedirs(gadm_grid_dir, exist_ok=True)
    os.makedirs(os.path.join(gadm_grid_dir, "countries_grid.zarr"))

    targets = cc.plan(PipelineStep.GRID, TargetSelection())
    assert len(targets) == 1
    assert targets[0].output_path == os.path.join(
        cc.output_root(PipelineStep.GRID), "classifications_grid.zarr"
    )


def test_country_classifications_requires_gadm_grid():
    spec = registry.resolve("country_classifications")
    assert spec.requires == (("gadm", PipelineStep.GRID),)
