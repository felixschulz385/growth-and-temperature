"""Grid-shake selection/normalisation and its integer-native-pixel realisation
in the DuckDB engine."""

import pytest

from src.data.assemble.config import validate_assembly_config
from src.data.assemble.grid_shake import (
    DEFAULT_GRID_SHAKE_PRESETS,
    normalize_grid_shake_offsets,
    resolve_shake_selection,
)
from src.data.assemble.sql_engine import GridFacts

from tests.data.assemble.conftest import grid, run_create, write_tiled_source


# --- normalize_grid_shake_offsets ----------------------------------------------


def test_normalize_grid_shake_offsets_none_or_empty():
    assert normalize_grid_shake_offsets(None) == []
    assert normalize_grid_shake_offsets({}) == []
    assert normalize_grid_shake_offsets([]) == []


def test_normalize_grid_shake_offsets_preset():
    specs = normalize_grid_shake_offsets("quad")
    assert [(dx, dy) for _, dx, dy in specs] == DEFAULT_GRID_SHAKE_PRESETS["quad"]
    assert [label for label, _, _ in specs] == ["0", "1", "2"]


def test_normalize_grid_shake_offsets_dict_and_list_forms():
    assert normalize_grid_shake_offsets({"offsets": [(0.5, 0.0)]}) == [("0", 0.5, 0.0)]
    assert normalize_grid_shake_offsets([(0.25, 0.25), (0.75, 0.75)]) == [
        ("0", 0.25, 0.25), ("1", 0.75, 0.75),
    ]


def test_normalize_grid_shake_offsets_out_of_range_raises():
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        normalize_grid_shake_offsets([(1.0, 0.0)])
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        normalize_grid_shake_offsets([(-0.1, 0.0)])


def test_normalize_grid_shake_offsets_unknown_preset_raises():
    with pytest.raises(ValueError, match="Unknown grid_shake preset"):
        normalize_grid_shake_offsets("not-a-real-preset")


# --- resolve_shake_selection -------------------------------------------------


def test_resolve_shake_selection_none_is_base_only():
    assert resolve_shake_selection(None) == [("base", 0.0, 0.0)]
    assert resolve_shake_selection("none") == [("base", 0.0, 0.0)]


def test_resolve_shake_selection_preset_is_base_plus_offsets():
    sel = resolve_shake_selection("quad")
    assert [label for label, _, _ in sel] == ["base", "s0", "s1", "s2"]
    assert [(dx, dy) for _, dx, dy in sel[1:]] == DEFAULT_GRID_SHAKE_PRESETS["quad"]


def test_resolve_shake_selection_single_offset_label():
    assert resolve_shake_selection("s1") == [("s1", 0.0, 0.5)]


def test_resolve_shake_selection_unknown_raises():
    with pytest.raises(ValueError, match="Unknown --shake value"):
        resolve_shake_selection("wobble")
    with pytest.raises(ValueError, match="out of range"):
        resolve_shake_selection("s9")


# --- config validation ------------------------------------------------------


def _base_assembly_config(**processing_overrides):
    return {
        "output_path": "/tmp/does-not-matter",
        "datasets": {"src": {"path": "/tmp/does-not-exist.zarr"}},
        "processing": {"resolution": 5000.0, **processing_overrides},
    }


def test_validate_assembly_config_accepts_valid_grid_shake():
    errors = validate_assembly_config(_base_assembly_config(grid_shake="quad"))
    assert not any("grid_shake" in e for e in errors)


def test_validate_assembly_config_rejects_bad_grid_shake():
    errors = validate_assembly_config(_base_assembly_config(grid_shake="nonsense"))
    assert any("grid_shake" in e for e in errors)


# --- integer-native-pixel realisation -------------------------------------------


def test_shake_fraction_becomes_integer_native_pixel_offset():
    assert GridFacts.build(5000.0, (0.5, 0.5), 2048).DR == 2   # round(0.5 * 5)
    assert GridFacts.build(10000.0, (0.5, 0.0), 2048) == GridFacts(
        W=34736, H=12704, TS=2048, F=10, DR=0, DC=5
    )
    # native grid: shake is a no-op
    assert GridFacts.build(None, (0.5, 0.5), 2048).DR == 0


def test_shake_variant_produces_a_different_but_same_schema_table(tmp_path):
    root = str(tmp_path / "g")
    src = write_tiled_source(
        root, "modis", W=20, H=16, years=[2000],
        value_fn=lambda r, c, y: {"lst_mean": float(r * 7 + c * 3)},
    )
    datasets = {"modis": {"path": src, "index_cols": ["pixel_id", "year"], "resampling": "average"}}

    base = run_create(grid(F=4), datasets, str(tmp_path / "base"))
    shaken = run_create(grid(F=4, DR=2, DC=2), datasets, str(tmp_path / "s0"),
                        shake_offset=(0.5, 0.5))

    assert list(base.columns) == list(shaken.columns)
    common = base.merge(shaken, on=["pixel_id", "year"], suffixes=("_b", "_s"), how="inner")
    assert len(common) > 0
    assert (common["lst_mean_b"] != common["lst_mean_s"]).any()
