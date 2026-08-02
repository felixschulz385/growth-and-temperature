"""Tests for the layout-aware default_mapping_path() -- see docs/design/09-
integrated-pipeline.md §14's deferred "layout: v2" task.
"""

from pathlib import Path

from src.analysis.subsets.registry import (
    DEFAULT_CLASSIFICATIONS_PATH,
    DEFAULT_COUNTRY_MAPPING_PATH,
    V2_CLASSIFICATIONS_PATH,
    V2_COUNTRY_MAPPING_PATH_TEMPLATE,
    default_classifications_path,
    default_mapping_path,
)


def test_default_mapping_path_uses_legacy_by_default():
    assert default_mapping_path(Path("/root")) == Path("/root") / DEFAULT_COUNTRY_MAPPING_PATH


def test_default_mapping_path_uses_v2_when_requested():
    assert default_mapping_path(Path("/root"), layout="v2") == Path("/root") / V2_COUNTRY_MAPPING_PATH_TEMPLATE.format(
        grid_id="legacy_4326"
    )


def test_default_mapping_path_v2_folds_grid_id_into_path():
    assert default_mapping_path(Path("/root"), layout="v2", grid_id="ease6933") == Path(
        "/root"
    ) / "data_nobackup/grid/ease6933/country_code_mapping.json"


def test_default_classifications_path_uses_legacy_by_default():
    assert default_classifications_path(Path("/root")) == Path("/root") / DEFAULT_CLASSIFICATIONS_PATH


def test_default_classifications_path_uses_v2_when_requested():
    assert default_classifications_path(Path("/root"), layout="v2") == Path("/root") / V2_CLASSIFICATIONS_PATH
