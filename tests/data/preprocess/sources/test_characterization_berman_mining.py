"""Characterization tests for the *current* BermanMiningPreprocessor
(pre-migration). docs/design/09-integrated-pipeline.md §10 step 0.

Note (correcting an earlier, unverified planning assumption): berman_mining
does NOT read any GADM output -- it only shares the VIIRS-derived geobox
cache location with osm/gadm (`get_or_create_geobox`, which builds the cache
independently from a VIIRS download if missing, not from GADM). No
`REQUIRES` edge on gadm is declared in the migration.
"""

import os

from src.data.preprocess.sources.berman_mining import BermanMiningPreprocessor


def _make_preprocessor(tmp_path, year_range=None, mining_data_path=None):
    kwargs = dict(hpc_target=str(tmp_path))
    if year_range is not None:
        kwargs["year_range"] = year_range
    if mining_data_path is not None:
        kwargs["mining_data_path"] = mining_data_path
    return BermanMiningPreprocessor(**kwargs)


def test_only_spatial_stage_is_supported(tmp_path):
    import pytest

    with pytest.raises(ValueError):
        BermanMiningPreprocessor(hpc_target=str(tmp_path), stage="annual")


def test_default_mining_data_path(tmp_path):
    pp = _make_preprocessor(tmp_path)
    assert pp.mining_data_path == os.path.join(str(tmp_path), "berman_mining", "raw", "baseline", "BCRT_baseline.dta")


def test_hpc_output_path(tmp_path):
    pp = _make_preprocessor(tmp_path)
    assert pp.get_hpc_output_path("spatial") == os.path.join(str(tmp_path), "berman_mining", "processed", "stage_2")


def test_spatial_target(tmp_path):
    pp = _make_preprocessor(tmp_path, year_range=[2000, 2010])
    targets = pp.get_preprocessing_targets("spatial")
    assert len(targets) == 1
    assert targets[0]["output_path"] == f"{pp.get_hpc_output_path('spatial')}/berman_mining_timeseries_reprojected.zarr"
    assert targets[0]["metadata"]["year_range"] == [2000, 2010]


def test_duplicate_get_hpc_output_path_and_from_config_definitions(tmp_path):
    # BUG (pinned, see class source): get_hpc_output_path and from_config are
    # each defined twice in the class body -- the second definitions silently
    # shadow the first. Both are byte-identical here so behaviour is
    # unaffected, but the dead first definitions are deleted in the migration
    # (docs/design/09-integrated-pipeline.md §5).
    import inspect

    source = inspect.getsource(BermanMiningPreprocessor)
    assert source.count("def get_hpc_output_path") == 2
    assert source.count("def from_config") == 2
