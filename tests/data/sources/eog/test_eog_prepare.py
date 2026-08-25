"""EogSource._execute_prepare: lazy per-tile clip (sel_bbox), routed through
Dask's worker processes (run_tiled_prepare_dask_year_major) instead of the
old single-process serial loop -- see
docs/design/13-prepare-memory-parallelism.md. A `.gz`-wrapped source's
decompressed temp file is deliberately left on disk rather than cleaned up
(no safe point to delete it under concurrent worker access -- see that
module's docstring), so this test checks it's written with a PID-unique
name under the source's own temp_dir rather than asserting it gets removed.
"""

import gzip
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import rasterio
from odc.geo.geobox import GeoBox
from rasterio.transform import from_bounds

from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources.eog.source import EogSource
from src.data.sources.steps import PipelineStep, TargetSelection, marker_path

_BASE_URLS = {"dmsp": "https://eogdata.mines.edu/wwwdata/dmsp/v4composites_rearrange/"}
_DATA_PATHS = {"dmsp": "eog/dmsp"}


@pytest.fixture(autouse=True)
def _no_real_eog_credentials_file(monkeypatch, tmp_path):
    from src.data.sources.eog import credentials as eog_credentials

    monkeypatch.setattr(eog_credentials, "DEFAULT_CREDENTIALS_PATH", tmp_path / "unused-eog-credentials.json")


def _make_source(tmp_path, source_type="dmsp", **raw):
    ctx = PipelineContext(data_root=str(tmp_path / "data_root"), local_index_dir=str(tmp_path / "index"))
    cfg = SourceConfig.from_dict(
        f"eog_{source_type}",
        {
            "data_path": _DATA_PATHS[source_type],
            "base_url": _BASE_URLS[source_type],
            # nearest, not the source's own "sum" default, so per-pixel
            # values pass through unchanged and this test can assert exact
            # values instead of reasoning about resampling arithmetic.
            "resampling": "nearest",
            **raw,
        },
    )
    return EogSource(ctx, cfg), ctx


def _write_tif(path, value, bounds=(-1, -1, 1, 1), size=8):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    transform = from_bounds(*bounds, size, size)
    with rasterio.open(
        path, "w", driver="GTiff", height=size, width=size, count=1,
        dtype="float32", crs="EPSG:4326", transform=transform,
    ) as dst:
        dst.write(np.full((size, size), value, dtype="float32"), 1)


def test_execute_prepare_clips_per_tile_and_writes_real_output(tmp_path):
    source, ctx = _make_source(tmp_path, year_range=[2020, 2020])
    raw_root = source.output_root(PipelineStep.FETCH)
    _write_tif(os.path.join(raw_root, "eog_dmsp_2020.tif"), 5.0)

    fake_geobox = GeoBox.from_bbox((-1, -1, 1, 1), crs="EPSG:4326", resolution=0.5)  # 4x4, 2x2 tiles @ size 2
    import src.data.common.geobox as geobox_module
    from unittest.mock import patch

    source.tile_size = 2

    targets = source.plan(PipelineStep.PREPARE, TargetSelection())
    assert len(targets) == 1
    target = targets[0]

    with patch.object(geobox_module, "get_target_geobox", lambda ctx: fake_geobox):
        assert source._execute_prepare(target) is True
    assert os.path.exists(marker_path(target.output_path))

    import pandas as pd

    parts = sorted(Path(target.output_path).glob("ix=*/iy=*/part-*.parquet"))
    assert len(parts) == 4  # 2x2 tile grid x 1 year
    df = pd.concat(pd.read_parquet(p) for p in parts)
    assert set(df["year"].unique()) == {2020}
    assert (df["dmsp"] == 5.0).all()


def test_execute_prepare_handles_gz_wrapped_source(tmp_path):
    source, ctx = _make_source(tmp_path, year_range=[2020, 2020])
    raw_root = source.output_root(PipelineStep.FETCH)
    tif_path = os.path.join(raw_root, "eog_dmsp_2020.tif")
    _write_tif(tif_path, 7.0)
    gz_path = tif_path + ".gz"
    with open(tif_path, "rb") as f_in, gzip.open(gz_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(tif_path)

    fake_geobox = GeoBox.from_bbox((-1, -1, 1, 1), crs="EPSG:4326", resolution=0.5)
    import src.data.common.geobox as geobox_module
    from unittest.mock import patch

    source.tile_size = 2

    targets = source.plan(PipelineStep.PREPARE, TargetSelection())
    target = targets[0]

    with patch.object(geobox_module, "get_target_geobox", lambda ctx: fake_geobox):
        assert source._execute_prepare(target) is True

    import pandas as pd

    parts = sorted(Path(target.output_path).glob("ix=*/iy=*/part-*.parquet"))
    df = pd.concat(pd.read_parquet(p) for p in parts)
    assert (df["dmsp"] == 7.0).all()

    # Decompressed to a fresh, PID-unique temp path under the source's own
    # temp_dir per worker that touched this year (deliberately not cleaned
    # up -- see module docstring in
    # src/data/common/prepare/raster_year_parallel.py), not colliding with
    # the fixed `local_file[:-3]` path the old single-process code used to
    # write and then delete. Bounded by worker count (here: at most 4), not
    # by tile count.
    leftover = [f for f in os.listdir(source.temp_dir) if f.endswith(".tif")]
    assert 1 <= len(leftover) <= 4
    assert "eog_dmsp_2020.tif" not in leftover
