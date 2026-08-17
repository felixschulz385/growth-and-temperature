"""Tests for the registry's default artefact paths (grid-id-aware GADM
mapping sidecar, prepared/-rooted classifications table).
"""

from pathlib import Path

from src.analysis.subsets.registry import (
    CLASSIFICATIONS_PATH,
    COUNTRY_MAPPING_PATH_TEMPLATE,
    default_classifications_path,
    default_mapping_path,
)


def test_default_mapping_path_uses_legacy_grid_by_default():
    assert default_mapping_path(Path("/root")) == Path("/root") / COUNTRY_MAPPING_PATH_TEMPLATE.format(
        grid_id="legacy_4326"
    )


def test_default_mapping_path_is_grid_independent():
    # The mapping itself is grid-independent (a string-code -> integer-id
    # table, not pixel data) -- `grid_id` is accepted for call-site symmetry
    # only (gid_mapping_path()'s docstring, src/data/sources/misc/gadm.py).
    assert default_mapping_path(Path("/root"), grid_id="ease6933") == Path(
        "/root"
    ) / "data_nobackup/prepared/misc/adm/gadm/GID_0_code_mapping.json"
    assert default_mapping_path(Path("/root"), grid_id="ease6933") == default_mapping_path(
        Path("/root"), grid_id="legacy_4326"
    )


def test_default_classifications_path():
    assert default_classifications_path(Path("/root")) == Path("/root") / CLASSIFICATIONS_PATH
