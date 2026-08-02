"""Tests for `get_target_geobox`: the single branch point that decides
which grid a source reprojects/rasterizes onto for a given `ctx.grid_id`.

Before this helper existed, every source except MODIS ignored `ctx.grid_id`
entirely and always reprojected onto the legacy grid, even when the output
*directory* was named `stage_2_ease6933` -- these tests pin the branch so
that regression can't silently reappear.
"""

import os

import pytest

import src.data.common.geobox.target as target_module
from src.data.common.geobox.target import CANONICAL_GEOBOX_CACHE_FILENAME, get_target_geobox
from src.data.pipeline.context import PipelineContext


def _make_ctx(tmp_path, grid_id):
    return PipelineContext(data_root=str(tmp_path), grid_id=grid_id)


def test_get_target_geobox_returns_legacy_by_default(tmp_path, monkeypatch):
    ctx = _make_ctx(tmp_path, "legacy_4326")

    calls = []

    def fake_get_or_create_geobox(hpc_root):
        calls.append(hpc_root)
        return "legacy-geobox"

    monkeypatch.setattr(target_module, "get_or_create_geobox", fake_get_or_create_geobox)

    result = get_target_geobox(ctx)

    assert result == "legacy-geobox"
    assert calls == [str(tmp_path)]


def test_get_target_geobox_returns_canonical_for_ease6933(tmp_path, monkeypatch):
    ctx = _make_ctx(tmp_path, "ease6933")

    calls = []

    def fake_get_or_create_canonical_geobox(cache_path):
        calls.append(cache_path)
        return "canonical-geobox"

    monkeypatch.setattr(
        target_module, "get_or_create_canonical_geobox", fake_get_or_create_canonical_geobox
    )

    result = get_target_geobox(ctx)

    assert result == "canonical-geobox"
    assert calls == [os.path.join(str(tmp_path), CANONICAL_GEOBOX_CACHE_FILENAME)]


def test_canonical_cache_path_uses_data_root_not_misc_subdir(tmp_path, monkeypatch):
    """Guards against accidentally reusing the legacy `<data_root>/misc/
    processed/stage_1/misc/` cache convention for the canonical geobox --
    the two grids' caches must not collide or nest inside one another."""
    ctx = _make_ctx(tmp_path, "ease6933")

    captured = {}

    def fake_get_or_create_canonical_geobox(cache_path):
        captured["cache_path"] = cache_path
        return "canonical-geobox"

    monkeypatch.setattr(
        target_module, "get_or_create_canonical_geobox", fake_get_or_create_canonical_geobox
    )

    get_target_geobox(ctx)

    assert "misc" not in captured["cache_path"]
    assert captured["cache_path"] == os.path.join(str(tmp_path), "canonical_geobox.pkl")


def test_get_target_geobox_rejects_unknown_grid_id(tmp_path):
    with pytest.raises(ValueError):
        _make_ctx(tmp_path, "not_a_real_grid")
