"""CountryClassificationsSource._execute_prepare(): writes a small
GID_0-keyed parquet table instead of rasterizing (classification values are
constant within a country, never per-pixel -- see module docstring)."""

import json
import os

import pandas as pd

from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources import layout
from src.data.sources.misc.country_classifications import CountryClassificationsSource
from src.data.sources.misc.gadm import gid_mapping_path
from src.data.sources.steps import PipelineStep, TargetSelection


def _make_source(tmp_path, grid_id="legacy_4326"):
    ctx = PipelineContext(
        data_root=str(tmp_path / "data_root"),
        local_index_dir=str(tmp_path / "index"),
        grid_id=grid_id,
    )
    cfg = SourceConfig.from_dict("country_classifications", {})
    return CountryClassificationsSource(ctx, cfg), ctx


def _write_fake_raw_files(source):
    hdi_file, wb_file = source._raw_file("hdi"), source._raw_file("worldbank")
    os.makedirs(os.path.dirname(hdi_file), exist_ok=True)
    open(hdi_file, "w").close()
    open(wb_file, "w").close()


def test_execute_prepare_writes_gid0_keyed_parquet(tmp_path):
    source, ctx = _make_source(tmp_path)
    _write_fake_raw_files(source)

    # Phase 1 (joining HDI+WB) already done in a prior run -- pre-populate
    # the real, externally-read classifications.parquet intermediate so
    # _execute_prepare's resumability check skips straight to phase 2.
    vector_dir = source.output_root(PipelineStep.PREPARE, agg=layout.ADM_AGG)
    os.makedirs(vector_dir, exist_ok=True)
    classifications_parquet = os.path.join(vector_dir, "classifications.parquet")
    pd.DataFrame(
        [
            {"iso3": "USA", "HDI_HI": True, "WB_HI": True},
            {"iso3": "FRA", "HDI_HI": True, "WB_HI": False},
            {"iso3": "ZZZ", "HDI_HI": False, "WB_HI": False},  # not in gadm mapping
        ]
    ).to_parquet(classifications_parquet, index=False)

    mapping_path = gid_mapping_path(ctx.data_root, ctx.grid_id, "GID_0")
    os.makedirs(os.path.dirname(mapping_path), exist_ok=True)
    with open(mapping_path, "w") as f:
        json.dump({"USA": 5, "FRA": 9}, f)

    targets = source.plan(PipelineStep.PREPARE, TargetSelection())
    assert len(targets) == 1
    target = targets[0]

    assert source.execute(target) is True
    result = pd.read_parquet(target.output_path)
    assert list(result.columns) == ["GID_0", "HDI_HI", "WB_HI"]
    # ZZZ has no gadm GID_0 mapping entry -> dropped, not zeroed.
    assert set(zip(result["GID_0"], result["HDI_HI"], result["WB_HI"])) == {
        (5, True, True),
        (9, True, False),
    }


def test_execute_prepare_fails_clearly_when_mapping_missing(tmp_path):
    source, ctx = _make_source(tmp_path)
    _write_fake_raw_files(source)

    vector_dir = source.output_root(PipelineStep.PREPARE, agg=layout.ADM_AGG)
    os.makedirs(vector_dir, exist_ok=True)
    classifications_parquet = os.path.join(vector_dir, "classifications.parquet")
    pd.DataFrame([{"iso3": "USA", "HDI_HI": True}]).to_parquet(classifications_parquet, index=False)
    # No GID_0_code_mapping.json written.

    targets = source.plan(PipelineStep.PREPARE, TargetSelection())
    assert len(targets) == 1
    assert source.execute(targets[0]) is False
