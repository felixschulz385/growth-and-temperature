"""OsmSource/GadmSource/CountryClassificationsSource.plan() must reproduce
the relevant slice of the old MiscPreprocessor's targets.
Oracle: tests/data/preprocess/sources/test_characterization_misc.py.
"""

import os

from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources import registry
from src.data.sources.steps import MissingPrerequisiteError, PipelineStep, TargetSelection


def _ctx(tmp_path, layout="legacy"):
    return PipelineContext(
        data_root=str(tmp_path / "data_root"), local_index_dir=str(tmp_path / "index"), layout=layout
    )


def _make(tmp_path, source_id, layout="legacy", **raw):
    ctx = _ctx(tmp_path, layout=layout)
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


def test_osm_gadm_country_classifications_fetch_and_prepare_use_top_level_trees_under_layout_v2(tmp_path):
    osm, ctx = _make(tmp_path, "osm", layout="v2")
    gadm, _ = _make(tmp_path, "gadm", layout="v2")
    cc, _ = _make(tmp_path, "country_classifications", layout="v2")

    assert osm.output_root(PipelineStep.FETCH) == os.path.join(ctx.data_root, "raw", "misc", "osm")
    assert osm.output_root(PipelineStep.PREPARE) == os.path.join(ctx.data_root, "prepared", "misc", "osm")
    assert gadm.output_root(PipelineStep.FETCH) == os.path.join(ctx.data_root, "raw", "misc", "gadm")
    assert gadm.output_root(PipelineStep.PREPARE) == os.path.join(ctx.data_root, "prepared", "misc", "gadm")
    assert cc.output_root(PipelineStep.FETCH) == os.path.join(ctx.data_root, "raw", "misc", "country_classifications")
    assert cc.output_root(PipelineStep.PREPARE) == os.path.join(
        ctx.data_root, "prepared", "misc", "country_classifications"
    )


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


def test_gadm_grid_target_includes_all_present_levels(tmp_path):
    gadm, _ = _make(tmp_path, "gadm")
    vector_dir = gadm.output_root(PipelineStep.PREPARE)
    os.makedirs(vector_dir, exist_ok=True)
    open(os.path.join(vector_dir, "gadm_levelADM_0_simplified.gpkg"), "w").close()

    targets = gadm.plan(PipelineStep.GRID, TargetSelection())
    assert len(targets) == 1
    assert len(targets[0].inputs) == 1  # only ADM_0 present so far

    open(os.path.join(vector_dir, "gadm_levelADM_1_simplified.gpkg"), "w").close()
    open(os.path.join(vector_dir, "gadm_levelADM_2_simplified.gpkg"), "w").close()
    targets = gadm.plan(PipelineStep.GRID, TargetSelection())
    assert len(targets[0].inputs) == 3
    # Sorted by level number, not filesystem/glob order.
    assert [os.path.basename(p) for p in targets[0].inputs] == [
        "gadm_levelADM_0_simplified.gpkg",
        "gadm_levelADM_1_simplified.gpkg",
        "gadm_levelADM_2_simplified.gpkg",
    ]


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
        cc.output_root(PipelineStep.GRID), "classifications_by_gid0.parquet"
    )


def test_country_classifications_requires_gadm_grid():
    spec = registry.resolve("country_classifications")
    assert spec.requires == (("gadm", PipelineStep.GRID),)


def test_osm_grid_target_uses_v2_family_path_under_layout_v2(tmp_path):
    osm, ctx = _make(tmp_path, "osm", layout="v2")
    vector_dir = osm.output_root(PipelineStep.PREPARE)
    os.makedirs(vector_dir, exist_ok=True)
    open(os.path.join(vector_dir, "land_polygons_simplified.gpkg"), "w").close()

    targets = osm.plan(PipelineStep.GRID, TargetSelection())
    assert len(targets) == 1
    assert targets[0].output_path == os.path.join(ctx.data_root, "grid", "legacy_4326", "land_mask.zarr")


def test_gadm_grid_target_uses_v2_family_path_under_layout_v2(tmp_path):
    gadm, ctx = _make(tmp_path, "gadm", layout="v2")
    vector_dir = gadm.output_root(PipelineStep.PREPARE)
    os.makedirs(vector_dir, exist_ok=True)
    open(os.path.join(vector_dir, "gadm_levelADM_0_simplified.gpkg"), "w").close()

    targets = gadm.plan(PipelineStep.GRID, TargetSelection())
    assert len(targets) == 1
    assert targets[0].output_path == os.path.join(ctx.data_root, "grid", "legacy_4326", "country_id.zarr")


def test_country_classifications_grid_finds_gadm_v2_output_under_layout_v2(tmp_path):
    # country_classifications' own GRID output is a small per-GID_0 parquet
    # table now, not a `<family>.zarr` pixel-grid store, so it doesn't
    # participate in layout:v2's shared grid/<grid_id>/ directory -- it falls
    # back to the legacy per-source path shape regardless of ctx.layout (see
    # module docstring / gadm.grid_store_path()'s v2_family=None fallback).
    # GADM's own dependency lookup (which *is* a v2 zarr family) is unaffected.
    cc, ctx = _make(tmp_path, "country_classifications", layout="v2")
    vector_dir = cc.output_root(PipelineStep.PREPARE)
    os.makedirs(vector_dir, exist_ok=True)
    open(os.path.join(vector_dir, "classifications.parquet"), "w").close()

    # GADM's v2 output not yet present -> no target.
    assert cc.plan(PipelineStep.GRID, TargetSelection()) == []

    grid_dir = os.path.join(ctx.data_root, "grid", "legacy_4326")
    os.makedirs(grid_dir, exist_ok=True)
    os.makedirs(os.path.join(grid_dir, "country_id.zarr"))

    targets = cc.plan(PipelineStep.GRID, TargetSelection())
    assert len(targets) == 1
    assert targets[0].output_path == os.path.join(
        ctx.data_root, "misc", "processed", "stage_2", "country_classifications",
        "classifications_by_gid0.parquet",
    )
    assert targets[0].inputs == (
        os.path.join(vector_dir, "classifications.parquet"),
        os.path.join(grid_dir, "country_id.zarr"),
    )
