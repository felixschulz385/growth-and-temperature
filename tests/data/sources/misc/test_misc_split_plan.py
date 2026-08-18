"""OsmSource/GadmSource/CountryClassificationsSource.plan() must reproduce
the relevant slice of the old MiscPreprocessor's targets.
Oracle: tests/data/preprocess/sources/test_characterization_misc.py.
"""

import os

from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources import layout, registry
from src.data.sources.steps import MissingPrerequisiteError, PipelineStep, TargetSelection


def _ctx(tmp_path):
    return PipelineContext(
        data_root=str(tmp_path / "data_root"), local_index_dir=str(tmp_path / "index")
    )


def _make(tmp_path, source_id, **raw):
    ctx = _ctx(tmp_path)
    cfg = SourceConfig.from_dict(source_id, dict(raw))
    cls = registry.load(source_id)
    return cls(ctx, cfg), ctx


def test_osm_gadm_country_classifications_fetch_and_prepare_use_top_level_trees(tmp_path):
    osm, ctx = _make(tmp_path, "osm")
    gadm, _ = _make(tmp_path, "gadm")
    cc, _ = _make(tmp_path, "country_classifications")

    assert osm.output_root(PipelineStep.FETCH) == os.path.join(ctx.data_root, "raw", "misc", "osm")
    assert osm.output_root(PipelineStep.PREPARE, agg=layout.MISC_AGG) == os.path.join(
        ctx.data_root, "prepared", "misc", "misc", "osm"
    )
    assert gadm.output_root(PipelineStep.FETCH) == os.path.join(ctx.data_root, "raw", "misc", "gadm")
    assert gadm.output_root(PipelineStep.PREPARE, agg=layout.ADM_AGG) == os.path.join(
        ctx.data_root, "prepared", "misc", "adm", "gadm"
    )
    assert cc.output_root(PipelineStep.FETCH) == os.path.join(ctx.data_root, "raw", "misc", "country_classifications")
    assert cc.output_root(PipelineStep.PREPARE, agg=layout.ADM_AGG) == os.path.join(
        ctx.data_root, "prepared", "misc", "adm", "country_classifications"
    )
    # Distinct index files -- the actual point of the split.
    assert osm.data_path == "misc/osm"
    assert gadm.data_path == "misc/gadm"
    assert osm.data_path != gadm.data_path


# osm/gadm's own PREPARE-target tests live in
# tests/data/sources/misc/test_osm_ledger_plan.py / test_gadm_ledger_plan.py --
# neither source declares a separate GRID step.


def test_country_classifications_prepare_target_tracks_which_sources_present(tmp_path):
    cc, _ = _make(tmp_path, "country_classifications")
    raw_dir = cc.output_root(PipelineStep.FETCH)
    os.makedirs(raw_dir, exist_ok=True)
    open(os.path.join(raw_dir, cc._raw_file("hdi").rsplit("/", 1)[-1]), "w").close()
    # Only HDI present.
    open(cc._raw_file("hdi"), "w").close()

    targets = cc.plan(PipelineStep.PREPARE, TargetSelection())
    assert len(targets) == 1
    assert targets[0].meta["has_hdi"] is True
    assert targets[0].meta["has_wb"] is False


def test_country_classifications_prepare_target_output_path(tmp_path):
    # country_classifications' PREPARE target's own output is at PREPARE's
    # path -- planning it doesn't need to probe gadm's output at all
    # (that's the runner's REQUIRES enforcement's job,
    # src/cli/data/handlers.py::_check_requires, gated before PREPARE runs).
    cc, _ = _make(tmp_path, "country_classifications")
    raw_dir = cc.output_root(PipelineStep.FETCH)
    os.makedirs(raw_dir, exist_ok=True)
    open(cc._raw_file("hdi"), "w").close()

    targets = cc.plan(PipelineStep.PREPARE, TargetSelection())
    assert len(targets) == 1
    assert targets[0].output_path == os.path.join(
        cc.output_root(PipelineStep.PREPARE, agg=layout.ADM_AGG), "classifications_by_gid0.parquet"
    )


def test_country_classifications_requires_gadm_prepare():
    # gadm's PREPARE builds its output directly; PipelineStep.GRID doesn't
    # exist for gadm to require, and country_classifications' own dependency
    # on it is scoped to its own PREPARE step.
    spec = registry.resolve("country_classifications")
    assert spec.requires == ((PipelineStep.PREPARE, "gadm", PipelineStep.PREPARE),)
    assert spec.requires_for(PipelineStep.FETCH) == ()
    assert spec.requires_for(PipelineStep.PREPARE) == (("gadm", PipelineStep.PREPARE),)


def test_osm_output_path_uses_family(tmp_path):
    osm, ctx = _make(tmp_path, "osm")
    assert osm._output_path() == os.path.join(osm.output_root(PipelineStep.GRID), "land_mask")


def test_gadm_output_path_uses_family(tmp_path):
    gadm, ctx = _make(tmp_path, "gadm")
    assert gadm._grid_output_path() == os.path.join(gadm.output_root(PipelineStep.GRID), "country_id")


def test_country_classifications_output_path_lives_under_prepare_root(tmp_path):
    # country_classifications' own output is a small per-GID_0 parquet table,
    # not a `<family>.zarr` pixel-grid store, so it lives under the PREPARE
    # root's ADM_AGG bucket rather than GRID's shared crs/<grid_id>/
    # directory (see module docstring). GADM's own output path (which *is* a
    # zarr family) is unaffected -- see test_gadm_output_path_uses_family
    # above.
    cc, ctx = _make(tmp_path, "country_classifications")
    assert cc._output_path() == os.path.join(
        cc.output_root(PipelineStep.PREPARE, agg=layout.ADM_AGG), "classifications_by_gid0.parquet"
    )
