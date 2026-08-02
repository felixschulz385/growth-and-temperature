"""Tests for the layout-aware default_mapping_path() -- see docs/design/09-
integrated-pipeline.md §14's deferred "layout: v2" task.
"""

from pathlib import Path

from src.analysis.subsets.registry import (
    DEFAULT_COUNTRY_MAPPING_PATH,
    V2_COUNTRY_MAPPING_PATH,
    default_mapping_path,
)


def test_default_mapping_path_uses_legacy_by_default():
    assert default_mapping_path(Path("/root")) == Path("/root") / DEFAULT_COUNTRY_MAPPING_PATH


def test_default_mapping_path_uses_v2_when_requested():
    assert default_mapping_path(Path("/root"), layout="v2") == Path("/root") / V2_COUNTRY_MAPPING_PATH
