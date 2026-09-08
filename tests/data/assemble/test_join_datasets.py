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
    # `_build_sources` normalizes `join_on` to a tuple of columns before it
    # reaches `_register_join_tables`; mirror that here so a bare-string spec
    # in a test reads naturally.
    norm = {
        n: ((jc,) if isinstance(jc, str) else tuple(jc), cfg)
        for n, (jc, cfg) in join_specs.items()
    }
    con = duckdb.connect()
    cols = se._register_join_tables(con, norm)
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


def test_register_composite_key_dedupes_on_full_tuple_and_excludes_both_keys(tmp_path):
    p = tmp_path / "plad.parquet"
    _write_join_table(p, [
        {"GID_2": 1, "year": 2000, "reg_fav": True},
        {"GID_2": 1, "year": 2001, "reg_fav": True},   # same GID_2, different year -> kept
        {"GID_2": 1, "year": 2001, "reg_fav": True},   # exact dup on (GID_2, year) -> dropped
    ])
    con, cols = _register(
        {"plad": (("GID_2", "year"), {"path": str(p), "join_on": ["GID_2", "year"]})}
    )
    assert cols["plad"] == (None, ["reg_fav"])          # neither join key is a value column
    assert con.sql("SELECT count(*) FROM join_plad").fetchone()[0] == 2


def test_register_composite_key_raises_when_any_column_absent(tmp_path):
    p = tmp_path / "plad.parquet"
    _write_join_table(p, [{"GID_2": 1, "reg_fav": True}])   # no `year`
    with pytest.raises(ValueError, match="no 'year' column"):
        _register({"plad": (("GID_2", "year"), {"path": str(p), "join_on": ["GID_2", "year"]})})


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


def test_composite_join_merges_by_gid_and_year(tmp_path):
    """PLAD-shaped (GID_N, year) sidecar: a pixel is favored only in the exact
    years its admin unit appears in the table; every other pixel-year -> False."""
    root = str(tmp_path / "g")
    # static per-pixel GID_2 (3 units), plus an annual source so the panel
    # carries a `year` column to key the composite join on.
    gadm = write_tiled_source(
        root, "gadm", W=W, H=H, years=None,
        value_fn=lambda r, c, y: {"GID_2": (r * W + c) % 3},
    )
    lst = write_tiled_source(
        root, "lst", W=W, H=H, years=[2000, 2001],
        value_fn=lambda r, c, y: {"lst": float(y)},
    )
    datasets = {
        "gadm": {"path": gadm, "index_cols": ["pixel_id"], "resampling": "mode"},
        "lst": {"path": lst, "index_cols": ["pixel_id", "year"], "resampling": "average"},
    }
    sidecar = str(tmp_path / "plad.parquet")
    # unit 1 favored in 2000 only; unit 2 favored in 2001 only; unit 0 never.
    _write_join_table(sidecar, [
        {"GID_2": 1, "year": 2000, "reg_fav": True},
        {"GID_2": 2, "year": 2001, "reg_fav": True},
    ])
    datasets["plad"] = {"path": sidecar, "join_on": ["GID_2", "year"], "fillna": False}

    df = run_create(grid(F=2), datasets, str(tmp_path / "out"))
    assert "reg_fav" in df.columns
    assert df["reg_fav"].notna().all()                       # fillna covered every miss
    favored = {tuple(x) for x in df.loc[df.reg_fav, ["GID_2", "year"]].to_numpy()}
    assert favored == {(1, 2000), (2, 2001)}


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


def test_validate_assembly_config_accepts_list_join_on(tmp_path):
    p = tmp_path / "plad.parquet"
    _write_join_table(p, [{"GID_2": 1, "year": 2000, "reg_fav": True}])
    config = {
        "output_path": str(tmp_path / "out"),
        "datasets": {"plad": {"path": str(p), "join_on": ["GID_2", "year"]}},
        "processing": {},
    }
    errors = validate_assembly_config(config)
    assert not any("join_on" in e for e in errors)


def test_validate_assembly_config_rejects_empty_list_join_on(tmp_path):
    p = tmp_path / "plad.parquet"
    _write_join_table(p, [{"GID_2": 1, "year": 2000, "reg_fav": True}])
    config = {
        "output_path": str(tmp_path / "out"),
        "datasets": {"plad": {"path": str(p), "join_on": []}},
        "processing": {},
    }
    errors = validate_assembly_config(config)
    assert any("join_on must be a non-empty string" in e for e in errors)
