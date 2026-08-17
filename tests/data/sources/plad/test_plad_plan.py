"""PlaDSource.plan()/execute() tests.

PLAD's PREPARE step doesn't rasterize: regional favoritism is constant
within a GID_1/GID_2 admin unit for a given year, so it writes a small
(GID_N, year)-keyed parquet table merged directly during assembly
(src.data.assemble.processors.TileProcessor's join_on mechanism) instead of
a pixel-grid zarr.
"""

import json
import os

import pandas as pd
import pytest

from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources.plad import PlaDSource
from src.data.sources.steps import PipelineStep, TargetSelection


def _make_source(tmp_path, admin_level=1, year_range=(1980, 2022), grid_id="legacy_4326"):
    ctx = PipelineContext(
        data_root=str(tmp_path / "data_root"),
        local_index_dir=str(tmp_path / "index"),
        grid_id=grid_id,
    )
    cfg = SourceConfig.from_dict("plad", {"admin_level": admin_level, "year_range": list(year_range)})
    return PlaDSource(ctx, cfg), ctx


def _write_gadm_mapping(ctx, gid_col, mapping):
    from src.data.sources.misc.gadm import gid_mapping_path

    path = gid_mapping_path(ctx.data_root, ctx.grid_id, gid_col)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(mapping, f)
    return path


def test_steps_is_fetch_and_prepare_only():
    assert PlaDSource.STEPS == (PipelineStep.FETCH, PipelineStep.PREPARE)


def test_requires_gadm_prepare():
    # PLAD doesn't rasterize, so it doesn't need GADM's polygon geometries --
    # only gadm's integer-id mapping sidecar, produced by gadm's PREPARE
    # step directly (PipelineStep.GRID doesn't exist anywhere). Scoped to
    # plad's own PREPARE step (its only step besides FETCH).
    assert PlaDSource.REQUIRES == ((PipelineStep.PREPARE, "gadm", PipelineStep.PREPARE),)


def test_output_root_hardcodes_plad_prefix_ignoring_data_path(tmp_path):
    ctx = PipelineContext(data_root=str(tmp_path / "data_root"), local_index_dir=str(tmp_path / "index"))
    cfg = SourceConfig.from_dict("plad", {"data_path": "something/else"})
    source = PlaDSource(ctx, cfg)
    # ADM_AGG: plad's PREPARE output is a GID_N-keyed admin table (module
    # docstring / src/data/sources/layout.py's crs/adm/misc split).
    assert source.output_root(PipelineStep.PREPARE) == os.path.join(ctx.data_root, "prepared", "plad", "adm")


def test_output_root_fetch_uses_top_level_tree(tmp_path):
    source, ctx = _make_source(tmp_path)
    assert source.output_root(PipelineStep.FETCH) == os.path.join(ctx.data_root, "raw", "plad")


def test_admin_level_must_be_1_or_2(tmp_path):
    ctx = PipelineContext(data_root=str(tmp_path / "data_root"), local_index_dir=str(tmp_path / "index"))
    cfg = SourceConfig.from_dict("plad", {"admin_level": 3})
    with pytest.raises(ValueError):
        PlaDSource(ctx, cfg)


def test_plan_prepare_empty_without_gadm_mapping_file(tmp_path):
    source, ctx = _make_source(tmp_path, admin_level=1)
    assert source.plan(PipelineStep.PREPARE, TargetSelection()) == []


def test_plan_prepare_target_output_path_includes_admin_level(tmp_path):
    s1, ctx1 = _make_source(tmp_path, admin_level=1)
    s2, ctx2 = _make_source(tmp_path, admin_level=2)
    _write_gadm_mapping(ctx1, "GID_1", {"USA.1_1": 1})
    _write_gadm_mapping(ctx2, "GID_2", {"USA.1.1_1": 1})

    t1 = s1.plan(PipelineStep.PREPARE, TargetSelection())[0]
    t2 = s2.plan(PipelineStep.PREPARE, TargetSelection())[0]
    assert t1.output_path == os.path.join(s1.output_root(PipelineStep.PREPARE), "plad_adm1_reg_fav.parquet")
    assert t2.output_path == os.path.join(s2.output_root(PipelineStep.PREPARE), "plad_adm2_reg_fav.parquet")
    assert t1.meta["admin_level"] == 1


def test_gid_mapping_file_path(tmp_path):
    source, ctx = _make_source(tmp_path, admin_level=1)
    # ADM_AGG, alongside gadm's own simplified `.gpkg` boundary files (see
    # gadm.gid_mapping_path()'s docstring).
    expected = os.path.join(ctx.data_root, "prepared", "misc", "adm", "gadm", "GID_1_code_mapping.json")
    assert source._gid_mapping_file() == expected


def test_build_reg_fav_table_translates_native_gid_to_gadm_integer_id(tmp_path, monkeypatch):
    source, ctx = _make_source(tmp_path, admin_level=1, year_range=(1990, 1996))

    plad_path = tmp_path / "plad_sample.dta"
    pd.DataFrame(
        {
            "gid_0": ["USA", "FRA"],
            "gid_1": ["USA.1_1", "FRA.2_1"],
            "gid_2": ["USA.1.1_1", "FRA.2.1_1"],
            "startyear": [1990, 1995],
            "endyear": [1992, 1996],
        }
    ).to_csv(plad_path, sep="\t", index=False)
    monkeypatch.setattr(source, "_resolve_plad_data_file", lambda: str(plad_path))

    mapping_file = _write_gadm_mapping(ctx, "GID_1", {"USA.1_1": 5, "FRA.2_1": 9})

    table = source._build_reg_fav_table(mapping_file)

    assert list(table.columns) == ["GID_1", "year", "reg_fav"]
    assert set(zip(table["GID_1"], table["year"])) == {
        (5, 1990), (5, 1991), (5, 1992),
        (9, 1995), (9, 1996),
    }
    assert table["reg_fav"].all()


def test_build_reg_fav_table_drops_codes_missing_from_gadm_mapping(tmp_path, monkeypatch):
    source, ctx = _make_source(tmp_path, admin_level=1, year_range=(1990, 1996))

    plad_path = tmp_path / "plad_sample.dta"
    pd.DataFrame(
        {
            "gid_0": ["USA"],
            "gid_1": ["USA.1_1"],
            "gid_2": ["USA.1.1_1"],
            "startyear": [1990],
            "endyear": [1990],
        }
    ).to_csv(plad_path, sep="\t", index=False)
    monkeypatch.setattr(source, "_resolve_plad_data_file", lambda: str(plad_path))

    # Mapping file exists but doesn't contain this row's GID_1 code.
    mapping_file = _write_gadm_mapping(ctx, "GID_1", {"OTHER.1_1": 1})

    table = source._build_reg_fav_table(mapping_file)
    assert table.empty


def test_execute_prepare_writes_reg_fav_parquet(tmp_path, monkeypatch):
    source, ctx = _make_source(tmp_path, admin_level=1, year_range=(1990, 1991))

    plad_path = tmp_path / "plad_sample.dta"
    pd.DataFrame(
        {
            "gid_0": ["USA"],
            "gid_1": ["USA.1_1"],
            "gid_2": ["USA.1.1_1"],
            "startyear": [1990],
            "endyear": [1991],
        }
    ).to_csv(plad_path, sep="\t", index=False)
    monkeypatch.setattr(source, "_resolve_plad_data_file", lambda: str(plad_path))

    _write_gadm_mapping(ctx, "GID_1", {"USA.1_1": 5})

    targets = source.plan(PipelineStep.PREPARE, TargetSelection())
    assert len(targets) == 1
    target = targets[0]

    assert source.execute(target) is True
    result = pd.read_parquet(target.output_path)
    assert set(zip(result["GID_1"], result["year"])) == {(5, 1990), (5, 1991)}
    assert result["reg_fav"].all()
