"""Tests for scripts/clean_legacy_grid_zarr.py -- the leftover pre-parquet
`<family>.zarr` GRID-store cleanup tool. Exercises the path-resolution and
find/delete logic against a fake tmp_path tree; no real HPC/SLURM involved.
"""

import os
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from clean_legacy_grid_zarr import (  # noqa: E402
    PIXEL_GRID_FAMILY,
    dir_size,
    human,
    legacy_zarr_path,
)

from src.data.pipeline.config import SourceConfig  # noqa: E402
from src.data.pipeline.context import PipelineContext  # noqa: E402
from src.data.sources import layout  # noqa: E402
from src.data.sources.acag import AcagSource  # noqa: E402
from src.data.sources.misc.gadm import GadmSource  # noqa: E402
from src.data.sources.eog.source import EogSource  # noqa: E402
from src.data.sources.glass.modis import GlassModisSource  # noqa: E402
from src.data.sources.modis.source import ModisSource  # noqa: E402


def _acag_source(tmp_path, grid_id="legacy_4326"):
    data_root = str(tmp_path / "data_root")
    cfg = SourceConfig.from_dict("acag", {"data_path": "acag/pm25"})
    return AcagSource(PipelineContext(data_root=data_root, grid_id=grid_id), cfg)


# -- non-pixel-grid sources are skipped entirely -----------------------------


def test_plad_and_country_classifications_and_commodity_prices_not_in_family_map():
    assert "plad" not in PIXEL_GRID_FAMILY
    assert "country_classifications" not in PIXEL_GRID_FAMILY
    assert "commodity_prices" not in PIXEL_GRID_FAMILY


# -- legacy_zarr_path() --------------------------------------------------------


def test_legacy_zarr_path_matches_the_current_path_with_zarr_suffix(tmp_path):
    source = _acag_source(tmp_path)
    path = legacy_zarr_path("acag", source, grid_id="legacy_4326")
    current = source._output_path()
    assert path == current + ".zarr"
    assert not current.endswith(".zarr")


def test_legacy_zarr_path_none_for_unmapped_source(tmp_path):
    source = _acag_source(tmp_path)
    assert legacy_zarr_path("plad", source, grid_id="legacy_4326") is None


def test_legacy_zarr_path_modis_forces_ease6933_regardless_of_grid_id(tmp_path):
    data_root = str(tmp_path / "data_root")
    cfg = SourceConfig.from_dict(
        "modis",
        {
            "product": "21A2",
            "day_range": {"start": [2019, 1], "end": [2019, 3]},
        },
    )
    source = ModisSource(PipelineContext(data_root=data_root, grid_id="legacy_4326"), cfg)
    path = legacy_zarr_path("modis", source, grid_id="legacy_4326")
    assert "/ease6933/" in path
    assert path.endswith("modis_lst_21a2.zarr")


def test_legacy_zarr_path_glass_modis_uses_variant_specific_family(tmp_path):
    data_root = str(tmp_path / "data_root")
    lst_cfg = SourceConfig.from_dict(
        "glass_modis",
        {"base_url": "https://x/", "day_range": {"start": [2019, 1], "end": [2019, 3]}, "land_tiles": ["h08v05"]},
    )
    ta_cfg = SourceConfig.from_dict(
        "glass_ta_modis",
        {"base_url": "https://x/", "day_range": {"start": [2019, 1], "end": [2019, 3]}, "land_tiles": ["h08v05"]},
    )
    lst_source = GlassModisSource(PipelineContext(data_root=data_root, grid_id="ease6933"), lst_cfg)
    ta_source = GlassModisSource(PipelineContext(data_root=data_root, grid_id="ease6933"), ta_cfg)

    lst_path = legacy_zarr_path("glass_modis", lst_source, grid_id="ease6933")
    ta_path = legacy_zarr_path("glass_ta_modis", ta_source, grid_id="ease6933")
    assert lst_path.endswith("glass_modis_lst.zarr")
    assert ta_path.endswith("glass_modis_ta.zarr")
    assert lst_path != ta_path


def test_legacy_zarr_path_eog_uses_source_type(tmp_path, monkeypatch):
    from src.data.sources.eog import credentials as eog_credentials

    monkeypatch.setattr(eog_credentials, "DEFAULT_CREDENTIALS_PATH", tmp_path / "unused-eog-credentials.json")

    data_root = str(tmp_path / "data_root")
    cfg = SourceConfig.from_dict(
        "eog_viirs",
        {"data_path": "eog/viirs", "base_url": "https://eogdata.mines.edu/nighttime_light/annual/v21/"},
    )
    source = EogSource(PipelineContext(data_root=data_root, grid_id="legacy_4326"), cfg)
    path = legacy_zarr_path("eog_viirs", source, grid_id="legacy_4326")
    assert path.endswith("eog_viirs_annual.zarr")


def test_legacy_zarr_path_gadm_uses_country_id_family(tmp_path):
    data_root = str(tmp_path / "data_root")
    cfg = SourceConfig.from_dict("gadm", {})
    source = GadmSource(PipelineContext(data_root=data_root, grid_id="legacy_4326"), cfg)
    path = legacy_zarr_path("gadm", source, grid_id="legacy_4326")
    assert path.endswith("country_id.zarr")


# -- dir_size() / human() ------------------------------------------------------


def test_dir_size_sums_nested_files(tmp_path):
    d = tmp_path / "pm25.zarr"
    d.mkdir()
    (d / "a.bin").write_bytes(b"x" * 10)
    (d / "sub").mkdir()
    (d / "sub" / "b.bin").write_bytes(b"y" * 5)
    assert dir_size(str(d)) == 15


def test_human_formats_reasonable_units():
    assert human(500) == "500.0B"
    assert human(2048) == "2.0KB"


# -- end-to-end: a leftover legacy zarr sits beside a real parquet-parts dir -


def test_leftover_zarr_found_new_parquet_parts_dir_untouched(tmp_path):
    source = _acag_source(tmp_path)

    legacy_path = legacy_zarr_path("acag", source, grid_id="legacy_4326")
    os.makedirs(legacy_path)
    Path(legacy_path, "zarr.json").write_text("{}")

    new_path = source._output_path()
    os.makedirs(os.path.join(new_path, "ix=0000", "iy=0000"), exist_ok=True)
    Path(new_path, "ix=0000", "iy=0000", "part.parquet").write_bytes(b"x")

    assert os.path.exists(legacy_path)
    assert os.path.exists(new_path)
    assert legacy_path != new_path

    import shutil

    shutil.rmtree(legacy_path)
    assert not os.path.exists(legacy_path)
    assert os.path.exists(new_path)  # untouched
