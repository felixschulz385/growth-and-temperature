"""join_on datasets: small GID-keyed tables merged onto the assembled panel by
an existing GID column, not block-aggregated onto the pixel grid."""

import duckdb
import pandas as pd
import pytest

from src.data.assemble import sql_engine as se
from src.data.assemble.config import validate_assembly_config

from tests.data.assemble.conftest import grid, run_create, write_tiled_source

W, H = 20, 16


def _write_join_table(path, rows):
    pd.DataFrame(rows).to_parquet(path, index=False)


def _register(join_specs):
    con = duckdb.connect()
    cols = se._register_join_tables(con, join_specs)
    return con, cols


# --- _register_join_tables --------------------------------------------------


def test_register_reads_table_and_reports_value_columns(tmp_path):
    p = tmp_path / "cc.parquet"
    _write_join_table(p, [{"GID_0": 1, "HDI_HI": True}, {"GID_0": 2, "HDI_HI": False}])
    con, cols = _register({"cc": ("GID_0", {"path": str(p), "join_on": "GID_0"})})
    assert cols["cc"] == (None, ["HDI_HI"])
    assert con.sql("SELECT count(*) FROM join_cc").fetchone()[0] == 2


def test_register_applies_column_prefix_but_not_to_join_key(tmp_path):
    p = tmp_path / "cc.parquet"
    _write_join_table(p, [{"GID_0": 1, "HDI_HI": True}])
    _con, cols = _register(
        {"cc": ("GID_0", {"path": str(p), "join_on": "GID_0", "column_prefix": "cc_"})}
    )
    assert cols["cc"] == (None, ["cc_HDI_HI"])


def test_register_dedupes_duplicate_join_keys_keeping_first(tmp_path):
    p = tmp_path / "cc.parquet"
    _write_join_table(p, [{"GID_0": 1, "value": "first"}, {"GID_0": 1, "value": "second"}])
    con, _cols = _register({"cc": ("GID_0", {"path": str(p), "join_on": "GID_0"})})
    got = con.sql("SELECT value FROM join_cc").fetchall()
    assert got == [("first",)]


def test_register_raises_when_join_column_absent(tmp_path):
    p = tmp_path / "cc.parquet"
    _write_join_table(p, [{"iso3": "USA", "HDI_HI": True}])
    with pytest.raises(ValueError, match="no 'GID_0' column"):
        _register({"cc": ("GID_0", {"path": str(p), "join_on": "GID_0"})})


# --- end-to-end join onto the panel ---------------------------------------------


def _panel_with_gid(tmp_path):
    root = str(tmp_path / "g")
    src = write_tiled_source(
        root, "gadm", W=W, H=H, years=None,
        value_fn=lambda r, c, y: {"GID_0": f"C{(r * W + c) % 3}"},
    )
    return {"gadm": {"path": src, "index_cols": ["pixel_id"], "resampling": "mode"}}


def test_join_merges_by_gid_and_fillna_covers_unmatched(tmp_path):
    datasets = _panel_with_gid(tmp_path)
    sidecar = str(tmp_path / "cc.parquet")
    # only C0 and C1 are in the sidecar; C2 rows must fill to False
    _write_join_table(sidecar, [{"GID_0": "C0", "HDI_HI": True}, {"GID_0": "C1", "HDI_HI": False}])
    datasets["cc"] = {"path": sidecar, "join_on": "GID_0", "fillna": False}

    df = run_create(grid(F=2), datasets, str(tmp_path / "out"))
    assert "HDI_HI" in df.columns
    by_gid = df.groupby("GID_0")["HDI_HI"].agg(lambda s: set(s))
    assert by_gid["C0"] == {True}
    assert by_gid["C1"] == {False}
    assert by_gid["C2"] == {False}   # unmatched -> fillna
    assert df["HDI_HI"].notna().all()


def test_join_without_fillna_leaves_unmatched_null(tmp_path):
    datasets = _panel_with_gid(tmp_path)
    sidecar = str(tmp_path / "cc.parquet")
    _write_join_table(sidecar, [{"GID_0": "C0", "score": 1.0}])
    datasets["cc"] = {"path": sidecar, "join_on": "GID_0"}

    df = run_create(grid(F=2), datasets, str(tmp_path / "out"))
    assert df.loc[df.GID_0 == "C0", "score"].notna().all()
    assert df.loc[df.GID_0 != "C0", "score"].isna().all()


# --- config validation (unchanged) --------------------------------------------


def test_validate_assembly_config_rejects_non_string_join_on(tmp_path):
    p = tmp_path / "cc.parquet"
    _write_join_table(p, [{"GID_0": 1, "HDI_HI": True}])
    config = {
        "output_path": str(tmp_path / "out"),
        "datasets": {"cc": {"path": str(p), "join_on": 123}},
        "processing": {},
    }
    errors = validate_assembly_config(config)
    assert any("join_on must be a non-empty string" in e for e in errors)


def test_validate_assembly_config_accepts_valid_join_on(tmp_path):
    p = tmp_path / "cc.parquet"
    _write_join_table(p, [{"GID_0": 1, "HDI_HI": True}])
    config = {
        "output_path": str(tmp_path / "out"),
        "datasets": {"cc": {"path": str(p), "join_on": "GID_0"}},
        "processing": {},
    }
    errors = validate_assembly_config(config)
    assert not any("join_on" in e for e in errors)
