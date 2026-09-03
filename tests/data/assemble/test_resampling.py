"""Per-variable resampling: `resolve_resampling` (name -> method) and the
`SQL_RESAMPLING_AGGREGATES` mapping the DuckDB engine downsamples each variable
with.
"""

import duckdb
import pytest

from src.data.assemble.constants import (
    DEFAULT_RESAMPLING_METHOD,
    SQL_RESAMPLING_AGGREGATES,
    VALID_RESAMPLING_METHODS,
)
from src.data.assemble.utils import resolve_resampling


# --- resolve_resampling ------------------------------------------------------


def test_none_config_is_default_for_every_var():
    assert resolve_resampling(None, ["a", "b"]) == {
        "a": DEFAULT_RESAMPLING_METHOD,
        "b": DEFAULT_RESAMPLING_METHOD,
    }


def test_string_config_applies_to_every_var():
    assert resolve_resampling("average", ["a", "b"]) == {"a": "average", "b": "average"}


def test_map_config_default_plus_glob_first_match_wins():
    cfg = {
        "default": "average",
        "valid_*count*": "sum",
        "*_sd": "min",
    }
    got = resolve_resampling(
        cfg,
        ["lst_day_mean", "lst_day_sd", "valid_period_count_annual", "valid_month_count_day_annual"],
    )
    assert got == {
        "lst_day_mean": "average",
        "lst_day_sd": "min",
        "valid_period_count_annual": "sum",
        "valid_month_count_day_annual": "sum",
    }


def test_map_config_without_default_falls_back_to_module_default():
    got = resolve_resampling({"*_count": "sum"}, ["x_count", "y_mean"])
    assert got == {"x_count": "sum", "y_mean": DEFAULT_RESAMPLING_METHOD}


def test_unknown_method_raises_even_with_no_var_names():
    with pytest.raises(ValueError, match="Unknown resampling method 'avg'"):
        resolve_resampling({"default": "avg"}, [])
    with pytest.raises(ValueError, match="Unknown resampling method"):
        resolve_resampling("bogus", ["a"])


def test_kernel_methods_are_rejected_now_that_downsampling_is_block_aggregation():
    # nearest/bilinear/cubic/... have no SQL block-aggregate equivalent.
    for m in ("nearest", "bilinear", "cubic", "lanczos", "gauss", "cubic_spline"):
        assert m not in VALID_RESAMPLING_METHODS
        with pytest.raises(ValueError, match="Unknown resampling method"):
            resolve_resampling(m, ["a"])


def test_non_str_non_mapping_config_raises():
    with pytest.raises(ValueError, match="must be a method string or a mapping"):
        resolve_resampling(["average"], ["a"])


# --- SQL_RESAMPLING_AGGREGATES -------------------------------------------------


def test_every_valid_method_has_a_sql_aggregate():
    assert set(VALID_RESAMPLING_METHODS) == set(SQL_RESAMPLING_AGGREGATES)


@pytest.mark.parametrize(
    "method,rows,expected",
    [
        ("average", [1.0, 2.0, 3.0], 2.0),
        ("sum", [1.0, 2.0, 3.0], 6.0),
        ("max", [1.0, 5.0, 3.0], 5.0),
        ("min", [1.0, 5.0, 3.0], 1.0),
        ("mode", [7.0, 7.0, 9.0], 7.0),
        ("med", [1.0, 2.0, 3.0, 4.0], 2.5),
    ],
)
def test_sql_aggregate_expression_computes_expected(method, rows, expected):
    con = duckdb.connect()
    con.execute("CREATE TABLE t(v DOUBLE)")
    con.executemany("INSERT INTO t VALUES (?)", [(r,) for r in rows])
    expr = SQL_RESAMPLING_AGGREGATES[method]("v")
    got = con.execute(f"SELECT {expr} FROM t").fetchone()[0]
    assert got == pytest.approx(expected)
