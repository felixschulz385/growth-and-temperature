"""Resolve a heterogeneous "data source" argument to a DuckDB relation.

Every ``src.viz.stats`` builder works against the SQL snippet returned here, so a
full panel never has to materialise in pandas. Accepted sources:

* a ``.parquet`` file, a glob, a partitioned tree (pointed at anywhere above
  the parts, e.g. the assembled panel's ``grid=*/shake=*/ix=*/iy=*/*.parquet``)
  or a flat directory of ``*.parquet``  -> ``read_parquet(..., union_by_name=true)``
  (the idiom used by ``src/data/assemble/sql_engine.py``);
* an already-open ``duckdb.DuckDBPyConnection`` plus ``table=<name>`` -- used as
  is and never closed;
* a pandas ``DataFrame`` -- registered as a temporary view for the lifetime of
  the context;
* a ``duckdb.DuckDBPyRelation`` -- driven through the connection that built it
  (a relation cannot be registered into any other connection).
"""

from __future__ import annotations

import contextlib
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import duckdb

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def quote_ident(name: str) -> str:
    """Validate ``name`` as a bare SQL identifier and return it double-quoted."""
    if not _IDENT_RE.fullmatch(name):
        raise ValueError(f"unsafe SQL identifier: {name!r}")
    return f'"{name}"'


class _RelationCursor:
    """Connection-like shim that runs SQL against a bare ``DuckDBPyRelation``
    through the relation's *own* connection.

    A relation is bound to the connection that built it and cannot be
    ``register``-ed anywhere else, so when a relation is passed without an
    explicit ``con=`` there is no connection object to hand back. This exposes
    just the ``execute(...)`` entry point the ``src.viz.stats`` builders use;
    ``DuckDBPyRelation.query`` returns a relation, which already supports
    ``fetchone`` / ``fetchall`` / ``fetchdf``.
    """

    def __init__(self, relation: "duckdb.DuckDBPyRelation", name: str):
        self._relation = relation
        self._name = name

    def execute(self, sql: str):
        return self._relation.query(self._name, sql)


@dataclass
class ResolvedSource:
    """A DuckDB connection (or a :class:`_RelationCursor`) plus a ``FROM``-able
    relation string."""

    con: "duckdb.DuckDBPyConnection | _RelationCursor"
    relation: str
    owns_con: bool


def _parquet_glob(path: Path) -> str:
    if not path.is_dir():
        return str(path)
    # A flat directory of parts, or a partitioned tree rooted right here.
    if any(path.glob("*.parquet")):
        return str(path / "*.parquet")
    # Otherwise the caller may have pointed above the partition columns (the
    # assembled panel lives at <root>/grid=*/shake=*/ix=*/iy=*/*.parquet);
    # recurse. `any(rglob(...))` stops at the first hit.
    if any(path.rglob("*.parquet")):
        return str(path / "**" / "*.parquet")
    # Nothing found -- keep the flat guess so the DuckDB error names a sane path.
    return str(path / "*.parquet")


@contextlib.contextmanager
def resolve_source(
    source,
    *,
    con: duckdb.DuckDBPyConnection | None = None,
    table: str | None = None,
) -> Iterator[ResolvedSource]:
    """Yield a :class:`ResolvedSource` for ``source`` (see module docstring)."""
    import pandas as pd

    if isinstance(source, duckdb.DuckDBPyConnection):
        if not table:
            raise ValueError("resolve_source: pass table=<name> alongside a duckdb connection")
        yield ResolvedSource(source, quote_ident(table), owns_con=False)
        return

    if isinstance(source, duckdb.DuckDBPyRelation):
        # Bound to its own connection; drive it there rather than trying to
        # register it into a fresh (or caller-supplied) connection.
        name = f"_viz_src_{uuid.uuid4().hex[:12]}"
        yield ResolvedSource(_RelationCursor(source, name), name, owns_con=False)
        return

    owns = con is None
    con = con or duckdb.connect()
    if owns:
        with contextlib.suppress(Exception):
            con.execute("SET enable_progress_bar = false")  # no widget spam in notebooks
    view: str | None = None
    try:
        if isinstance(source, pd.DataFrame):
            view = f"_viz_src_{uuid.uuid4().hex[:12]}"
            con.register(view, source)
            relation = view
        elif isinstance(source, (str, Path)):
            text = str(source)
            if any(ch in text for ch in "*?[") and not os.path.exists(text):
                relation = f"read_parquet('{text}', union_by_name=true)"
            else:
                relation = f"read_parquet('{_parquet_glob(Path(text))}', union_by_name=true)"
        else:
            raise TypeError(f"resolve_source: unsupported source type {type(source)!r}")

        yield ResolvedSource(con, relation, owns_con=owns)
    finally:
        if view is not None:
            with contextlib.suppress(Exception):
                con.unregister(view)
        if owns:
            con.close()


def columns(resolved: ResolvedSource) -> list[str]:
    """Column names of the resolved relation (via ``DESCRIBE ... LIMIT 0``)."""
    rows = resolved.con.execute(
        f"DESCRIBE SELECT * FROM {resolved.relation} LIMIT 0"
    ).fetchall()
    return [row[0] for row in rows]
