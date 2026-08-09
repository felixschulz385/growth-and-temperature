"""ModisSource.plan()/transfer_units() must reproduce the old
MODISPreprocessor's behaviour. Oracle: tests/data/preprocess/sources/test_characterization_modis.py.
"""

import os

from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources.modis.source import ModisSource
from src.data.sources.steps import PipelineStep, TargetSelection


def _make_source(tmp_path, tiles=("h18v04", "h20v08"), year_range=(2019, 2020), layout="legacy", **extra_raw):
    data_root = str(tmp_path / "data_root")
    local_index_dir = str(tmp_path / "index")
    ctx = PipelineContext(data_root=data_root, local_index_dir=local_index_dir, layout=layout)
    cfg = SourceConfig.from_dict("modis", {"year_range": list(year_range), "tiles": list(tiles), **extra_raw})
    return ModisSource(ctx, cfg), ctx


def test_fetch_and_grid_are_the_only_steps():
    assert ModisSource.STEPS == (PipelineStep.FETCH, PipelineStep.GRID)


def test_output_root_uses_ease6933_suffix_for_grid(tmp_path):
    source, ctx = _make_source(tmp_path)
    assert source.output_root(PipelineStep.FETCH) == os.path.join(ctx.data_root, "modis/21A2", "processed", "stage_1")
    assert source.output_root(PipelineStep.GRID) == os.path.join(ctx.data_root, "modis/21A2", "processed", "stage_2_ease6933")


def test_output_root_fetch_uses_top_level_tree_under_layout_v2(tmp_path):
    # FETCH is a rename of the old PREPARE ("annual") step -- the physical
    # artifact tree is unchanged, not layout.raw_root()'s bare <data_path>/raw
    # convention every crawler-based FETCH source uses (module docstring).
    source, ctx = _make_source(tmp_path, layout="v2")
    assert source.output_root(PipelineStep.FETCH) == os.path.join(ctx.data_root, "prepared", "modis/21A2")


def test_data_path_defaults_to_product_specific(tmp_path):
    source, _ = _make_source(tmp_path, product="11A1")
    assert source.cfg.data_path == "modis/11A1"


def test_fetch_targets_one_per_tile_year(tmp_path):
    source, _ = _make_source(tmp_path, tiles=("h18v04", "h20v08"), year_range=(2019, 2020))
    targets = source.plan(PipelineStep.FETCH, TargetSelection())
    assert {t.key for t in targets} == {"2019/h18v04", "2019/h20v08", "2020/h18v04", "2020/h20v08"}
    sample = next(t for t in targets if t.key == "2019/h18v04")
    assert sample.output_path == os.path.join(source.output_root(PipelineStep.FETCH), "2019", "h18v04.tif")


def test_fetch_targets_use_explicit_years_override_not_year_range(tmp_path):
    # modis_robustness_11a1's "3-5 years spanning early/mid/late mission"
    # (data.yaml) needs discrete, non-contiguous years -- year_range alone
    # can only express one contiguous span.
    source, _ = _make_source(tmp_path, tiles=("h18v04",), years=[2004, 2014, 2023])
    targets = source.plan(PipelineStep.FETCH, TargetSelection())
    assert {t.key for t in targets} == {"2004/h18v04", "2014/h18v04", "2023/h18v04"}


def test_grid_targets_one_per_year_with_stage1_output(tmp_path):
    source, _ = _make_source(tmp_path, year_range=(2019, 2020))
    stage1 = source.output_root(PipelineStep.FETCH)
    year_dir = os.path.join(stage1, "2019")
    os.makedirs(year_dir, exist_ok=True)
    open(os.path.join(year_dir, "h18v04.tif"), "w").close()

    targets = source.plan(PipelineStep.GRID, TargetSelection())
    assert len(targets) == 1
    assert targets[0].key == "2019"
    assert targets[0].inputs == (os.path.join(year_dir, "h18v04.tif"),)
    assert targets[0].output_path == os.path.join(
        source.output_root(PipelineStep.GRID), "modis_21A2_timeseries_reprojected.zarr"
    )


def test_grid_target_uses_v2_family_path_under_layout_v2(tmp_path):
    for product, family in (("21A2", "modis_lst_21a2"), ("11A1", "modis_lst_11a1")):
        source, ctx = _make_source(tmp_path, year_range=(2019, 2020), layout="v2", product=product)
        stage1 = source.output_root(PipelineStep.FETCH)
        year_dir = os.path.join(stage1, "2019")
        os.makedirs(year_dir, exist_ok=True)
        open(os.path.join(year_dir, "h18v04.tif"), "w").close()

        targets = source.plan(PipelineStep.GRID, TargetSelection())
        assert len(targets) == 1
        # MODIS forces grid_id=ease6933 unconditionally (see output_root()),
        # independent of ctx.grid_id -- so the v2 path reflects that too.
        assert targets[0].output_path == os.path.join(ctx.data_root, "grid", "ease6933", f"{family}.zarr")


def test_grid_targets_always_never_complete_the_quirk(tmp_path):
    from src.data.sources.steps import Completion, is_complete

    source, _ = _make_source(tmp_path, year_range=(2019, 2020))
    stage1 = source.output_root(PipelineStep.FETCH)
    year_dir = os.path.join(stage1, "2019")
    os.makedirs(year_dir, exist_ok=True)
    open(os.path.join(year_dir, "h18v04.tif"), "w").close()

    targets = source.plan(PipelineStep.GRID, TargetSelection())
    assert targets[0].completion is Completion.NEVER
    os.makedirs(targets[0].output_path, exist_ok=True)  # pretend the shared zarr already exists
    assert is_complete(targets[0]) is False


def test_grid_targets_use_explicit_years_override_not_year_range(tmp_path):
    source, _ = _make_source(tmp_path, tiles=("h18v04",), years=[2004, 2014])
    stage1 = source.output_root(PipelineStep.FETCH)
    for year in (2004, 2014):
        year_dir = os.path.join(stage1, str(year))
        os.makedirs(year_dir, exist_ok=True)
        open(os.path.join(year_dir, "h18v04.tif"), "w").close()

    targets = source.plan(PipelineStep.GRID, TargetSelection())
    assert {t.key for t in targets} == {"2004", "2014"}


def test_transfer_units_one_per_tile_year_file_for_fetch(tmp_path):
    source, _ = _make_source(tmp_path, year_range=(2019, 2020))
    stage1 = source.output_root(PipelineStep.FETCH)
    for year, tile in [("2019", "h18v04"), ("2019", "h20v08")]:
        d = os.path.join(stage1, year)
        os.makedirs(d, exist_ok=True)
        open(os.path.join(d, f"{tile}.tif"), "w").close()

    units = source.transfer_units(PipelineStep.FETCH)
    assert {u.unit_id for u in units} == {"2019/h18v04.tif", "2019/h20v08.tif"}


def test_transfer_units_grid_falls_back_to_single_unit_default(tmp_path):
    source, _ = _make_source(tmp_path, year_range=(2019, 2020))
    units = source.transfer_units(PipelineStep.GRID)
    assert len(units) == 1
    assert units[0].unit_id == "grid"


def test_modis_robustness_11a1_alias_resolves():
    # BUG (pinned, see module docstring): the old factory only matched
    # "modis"/"modis_lst"; "modis_robustness_11a1" raised ModuleNotFoundError.
    from src.data.sources import registry

    assert registry.resolve("modis_robustness_11a1").id == "modis"


def test_get_stac_client_is_cached_across_calls(tmp_path, monkeypatch):
    # _execute_fetch runs once per (year, tile) StepTarget -- an uncached
    # client would reopen the STAC catalog (an HTTP round trip) on every
    # single one instead of once per ModisSource instance. Fakes both
    # `pystac_client`/`planetary_computer` in sys.modules (imported lazily
    # inside `_get_stac_client()`) so this doesn't need either installed.
    import sys
    import types

    source, _ = _make_source(tmp_path)
    open_calls = []

    class _FakeClient:
        pass

    class _FakeClientCls:
        @staticmethod
        def open(url, modifier=None):
            open_calls.append(url)
            return _FakeClient()

    fake_pystac_client = types.ModuleType("pystac_client")
    fake_pystac_client.Client = _FakeClientCls
    fake_planetary_computer = types.ModuleType("planetary_computer")
    fake_planetary_computer.sign_inplace = lambda item: item

    monkeypatch.setitem(sys.modules, "pystac_client", fake_pystac_client)
    monkeypatch.setitem(sys.modules, "planetary_computer", fake_planetary_computer)

    client_a = source._get_stac_client()
    client_b = source._get_stac_client()
    assert client_a is client_b
    assert len(open_calls) == 1
