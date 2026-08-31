"""Tests for join_on datasets: small GID-keyed tables merged directly onto
assembled rows instead of being reprojected onto the pixel grid."""

import pandas as pd
import pytest

from src.data.assemble.config import validate_assembly_config
from src.data.assemble.processors import TileProcessor


def _base_assembly_config(tmp_path, **processing_overrides):
    return {
        "output_path": str(tmp_path / "out"),
        "datasets": {
            "gadm": {"path": str(tmp_path / "does-not-exist.zarr")},
        },
        "processing": {**processing_overrides},
    }


def _write_join_table(path, rows):
    pd.DataFrame(rows).to_parquet(path, index=False)


# --- _load_join_tables -------------------------------------------------


def test_load_join_tables_reads_small_table_keyed_by_gid(tmp_path):
    table_path = tmp_path / "classifications.parquet"
    _write_join_table(table_path, [{"GID_0": 1, "HDI_HI": True}, {"GID_0": 2, "HDI_HI": False}])

    config = {
        "output_path": str(tmp_path / "out"),
        "datasets": {
            "classifications": {"path": str(table_path), "join_on": "GID_0"},
        },
        "processing": {},
    }
    processor = TileProcessor(config, str(tmp_path / "out"))

    assert "classifications" in processor.join_tables
    join_col, table = processor.join_tables["classifications"]
    assert join_col == "GID_0"
    assert list(table["GID_0"]) == [1, 2]
    assert processor.column_order_map["classifications"] == ["HDI_HI"]


def test_load_join_tables_applies_column_prefix(tmp_path):
    table_path = tmp_path / "classifications.parquet"
    _write_join_table(table_path, [{"GID_0": 1, "HDI_HI": True}])

    config = {
        "output_path": str(tmp_path / "out"),
        "datasets": {
            "classifications": {
                "path": str(table_path), "join_on": "GID_0", "column_prefix": "cc_",
            },
        },
        "processing": {},
    }
    processor = TileProcessor(config, str(tmp_path / "out"))

    join_col, table = processor.join_tables["classifications"]
    assert join_col == "GID_0"  # join column itself is never prefixed
    assert "cc_HDI_HI" in table.columns
    assert processor.column_order_map["classifications"] == ["cc_HDI_HI"]


def test_load_join_tables_dedupes_duplicate_join_keys(tmp_path):
    table_path = tmp_path / "classifications.parquet"
    _write_join_table(
        table_path,
        [{"GID_0": 1, "value": "first"}, {"GID_0": 1, "value": "second"}],
    )

    config = {
        "output_path": str(tmp_path / "out"),
        "datasets": {"classifications": {"path": str(table_path), "join_on": "GID_0"}},
        "processing": {},
    }
    processor = TileProcessor(config, str(tmp_path / "out"))

    _, table = processor.join_tables["classifications"]
    assert len(table) == 1
    assert table.iloc[0]["value"] == "first"


def test_load_join_tables_skips_table_missing_join_column(tmp_path):
    table_path = tmp_path / "classifications.parquet"
    _write_join_table(table_path, [{"iso3": "USA", "HDI_HI": True}])

    config = {
        "output_path": str(tmp_path / "out"),
        "datasets": {"classifications": {"path": str(table_path), "join_on": "GID_0"}},
        "processing": {},
    }
    processor = TileProcessor(config, str(tmp_path / "out"))

    assert "classifications" not in processor.join_tables


# --- _apply_join_tables --------------------------------------------------


def test_apply_join_tables_merges_by_existing_gid_column(tmp_path):
    table_path = tmp_path / "classifications.parquet"
    _write_join_table(
        table_path,
        [{"GID_0": 1, "HDI_HI": True}, {"GID_0": 2, "HDI_HI": False}],
    )

    config = {
        "output_path": str(tmp_path / "out"),
        "datasets": {"classifications": {"path": str(table_path), "join_on": "GID_0"}},
        "processing": {},
    }
    processor = TileProcessor(config, str(tmp_path / "out"))

    combined = pd.DataFrame({"pixel_id": [10, 11, 12], "GID_0": [1, 2, 1]})
    result = processor._apply_join_tables(combined)

    assert list(result["HDI_HI"]) == [True, False, True]
    assert list(result["pixel_id"]) == [10, 11, 12]


def test_apply_join_tables_fills_missing_gid_with_fillna_config(tmp_path):
    table_path = tmp_path / "classifications.parquet"
    _write_join_table(table_path, [{"GID_0": 1, "HDI_HI": True}])

    config = {
        "output_path": str(tmp_path / "out"),
        "datasets": {
            "classifications": {
                "path": str(table_path), "join_on": "GID_0", "fillna": False,
            },
        },
        "processing": {},
    }
    processor = TileProcessor(config, str(tmp_path / "out"))

    combined = pd.DataFrame({"pixel_id": [10, 11], "GID_0": [1, 999]})
    result = processor._apply_join_tables(combined)

    assert list(result["HDI_HI"]) == [True, False]


def test_apply_join_tables_skips_when_gid_column_missing(tmp_path):
    table_path = tmp_path / "classifications.parquet"
    _write_join_table(table_path, [{"GID_0": 1, "HDI_HI": True}])

    config = {
        "output_path": str(tmp_path / "out"),
        "datasets": {"classifications": {"path": str(table_path), "join_on": "GID_0"}},
        "processing": {},
    }
    processor = TileProcessor(config, str(tmp_path / "out"))

    combined = pd.DataFrame({"pixel_id": [10, 11]})  # no GID_0 column at all
    result = processor._apply_join_tables(combined)

    assert "HDI_HI" not in result.columns
    assert list(result["pixel_id"]) == [10, 11]


# --- config validation -----------------------------------------------------


def test_validate_assembly_config_rejects_non_string_join_on(tmp_path):
    table_path = tmp_path / "classifications.parquet"
    _write_join_table(table_path, [{"GID_0": 1, "HDI_HI": True}])

    config = {
        "output_path": str(tmp_path / "out"),
        "datasets": {"classifications": {"path": str(table_path), "join_on": 123}},
        "processing": {},
    }
    errors = validate_assembly_config(config)
    assert any("join_on must be a non-empty string" in e for e in errors)


def test_validate_assembly_config_accepts_valid_join_on(tmp_path):
    table_path = tmp_path / "classifications.parquet"
    _write_join_table(table_path, [{"GID_0": 1, "HDI_HI": True}])

    config = {
        "output_path": str(tmp_path / "out"),
        "datasets": {"classifications": {"path": str(table_path), "join_on": "GID_0"}},
        "processing": {},
    }
    errors = validate_assembly_config(config)
    assert not any("join_on" in e for e in errors)


# --- loaders.load_all_datasets skips join_on datasets -----------------------


def test_load_all_datasets_skips_join_on_and_raises_clear_error_on_update(tmp_path):
    from src.data.assemble.loaders import load_all_datasets

    table_path = tmp_path / "classifications.parquet"
    _write_join_table(table_path, [{"GID_0": 1, "HDI_HI": True}])

    config = {
        "datasets": {"classifications": {"path": str(table_path), "join_on": "GID_0"}},
        "processing": {},
    }

    with pytest.raises(ValueError, match="No datasets could be loaded"):
        load_all_datasets(config, target_geobox=None)

    with pytest.raises(ValueError, match="join_on"):
        load_all_datasets(config, target_geobox=None, datasource_filter="classifications")
