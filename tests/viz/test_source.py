from __future__ import annotations

import duckdb
import pytest

from src.viz.source import columns, quote_ident, resolve_source

_EXPECTED = {"unit", "year", "country", "x", "treat", "y", "ntl"}


def _count(rs) -> int:
    return rs.con.execute(f"SELECT count(*) FROM {rs.relation}").fetchone()[0]


def test_resolve_flat_parquet_file(panel_parquet, panel_frame):
    with resolve_source(panel_parquet) as rs:
        assert _count(rs) == len(panel_frame)
        assert set(columns(rs)) == _EXPECTED


def test_resolve_flat_directory(panel_parquet, panel_frame, tmp_path):
    # panel_parquet lives alone in tmp_path -> point at the directory
    with resolve_source(str(tmp_path)) as rs:
        assert _count(rs) == len(panel_frame)


def test_resolve_hive_directory(panel_hive, panel_frame):
    with resolve_source(panel_hive) as rs:
        assert _count(rs) == len(panel_frame)
        assert "ix" in columns(rs) and "iy" in columns(rs)


def test_resolve_glob(panel_hive, panel_frame):
    with resolve_source(f"{panel_hive}/ix=*/iy=*/*.parquet") as rs:
        assert _count(rs) == len(panel_frame)


def test_resolve_dataframe(panel_frame):
    with resolve_source(panel_frame) as rs:
        assert _count(rs) == len(panel_frame)


def test_resolve_connection_and_table(panel_frame):
    con = duckdb.connect()
    con.register("v", panel_frame)
    con.execute("CREATE TABLE panel AS SELECT * FROM v")
    with resolve_source(con, table="panel") as rs:
        assert rs.owns_con is False
        assert _count(rs) == len(panel_frame)
    # a bare connection is never closed by resolve_source
    assert con.execute("SELECT count(*) FROM panel").fetchone()[0] == len(panel_frame)
    con.close()


def test_resolve_reuses_passed_connection(panel_parquet):
    con = duckdb.connect()
    with resolve_source(panel_parquet, con=con) as rs:
        assert rs.con is con
        assert rs.owns_con is False
    # still usable afterwards
    con.execute("SELECT 1").fetchone()
    con.close()


def test_resolve_connection_requires_table():
    con = duckdb.connect()
    with pytest.raises(ValueError):
        with resolve_source(con):
            pass
    con.close()


def test_quote_ident_rejects_injection():
    with pytest.raises(ValueError):
        quote_ident('x"; DROP TABLE t; --')
    assert quote_ident("ntl_harm") == '"ntl_harm"'
