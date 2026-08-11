"""DuckDB persistence for the regularized (typed, per-subsection-type)
detail-export tables. Schema-per-table is dynamic (inferred from the row
dicts each `regularize/subsections/*.py` module returns), following the
same dynamic-schema DuckDB-write pattern already used by
`snl_mining_manual_xls_to_duckdb.ipynb` and
`SnlMiningSource._export_admin_count_tables` elsewhere in this source --
column sets genuinely differ per subsection type, so there's no fixed DDL to
declare upfront.
"""

from __future__ import annotations

import datetime as _dt
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd


def _stabilize_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Resolve pandas `object`-dtype columns (text, `datetime.date`, or an
    all-`None` column of unknowable type) to an explicit, stable dtype
    before the DataFrame is used to infer or insert into a DuckDB table.

    Necessary because `CREATE TABLE IF NOT EXISTS ... AS SELECT * FROM df`
    infers each column's DuckDB type from *whichever mine's batch happens to
    create the table first* -- if that batch's value for a given column is
    entirely `None` (a real, common case: e.g. `millhead_grade_unit` is only
    populated for commodities that report a recovery-rate unit), DuckDB has
    been observed to resolve an all-null `object` column's type as `INT32`.
    A later mine's real string value for that same column then fails to
    insert (`Could not convert string ... to INT32`). Defaulting ambiguous
    columns to VARCHAR (pandas `"string"` dtype) is the safe choice: it's
    the widest type any later value -- numeric, date, or text -- can still
    be inserted into.
    """
    for column in df.columns:
        if df[column].dtype != object:
            continue
        non_null = df[column].dropna()
        if not non_null.empty and non_null.map(lambda v: isinstance(v, (_dt.date, _dt.datetime))).all():
            df[column] = pd.to_datetime(df[column])
        else:
            df[column] = df[column].astype("string")
    return df


def upsert_regularized_rows(conn, table_name: str, mine_id: str, rows: list[dict]) -> int:
    """Delete-then-insert *rows* (each already carrying a `mine_id` column,
    injected by the calling stage) into *table_name*, creating it on first
    use with a schema inferred from the row dicts. A no-op (returns 0) for
    an empty *rows* -- an empty DataFrame has no columns, so there's nothing
    to align against an existing table's schema, and creating a zero-column
    table on first use would be useless."""
    if not rows:
        return 0
    df = _stabilize_object_columns(pd.DataFrame(rows))
    conn.register("_new_regularized_rows", df)
    try:
        conn.execute(f'CREATE TABLE IF NOT EXISTS "{table_name}" AS SELECT * FROM _new_regularized_rows WHERE 0=1')
        conn.execute(f'DELETE FROM "{table_name}" WHERE mine_id = ?', [mine_id])
        conn.execute(f'INSERT INTO "{table_name}" SELECT * FROM _new_regularized_rows')
    finally:
        conn.unregister("_new_regularized_rows")
    return len(df)


def persist_regularized_tables(
    conn,
    mine_id: str,
    xls_sha256: str | None,
    tables: dict[str, list[dict]],
) -> dict[str, int]:
    """Inject the common audit columns (`mine_id`, `xls_sha256`,
    `regularized_at`) into every row of every table `regularize()` produced,
    then persist each via `upsert_regularized_rows`. Centralized here so
    individual subsection modules only need to return business fields."""
    regularized_at = datetime.now(timezone.utc)
    counts: dict[str, int] = {}
    for table_name, rows in tables.items():
        for row in rows:
            row.setdefault("mine_id", mine_id)
            row.setdefault("xls_sha256", xls_sha256)
            row.setdefault("regularized_at", regularized_at)
        counts[table_name] = upsert_regularized_rows(conn, table_name, mine_id, rows)
    return counts


def list_regularized_table_names(conn) -> list[str]:
    rows = conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_name LIKE 'detail_%' ORDER BY table_name"
    ).fetchall()
    return [row[0] for row in rows]


def export_regularized_tables_to_csv(
    conn,
    output_dir: str | Path,
    table_names: Iterable[str] | None = None,
) -> list[str]:
    """Dump each regularized table to `<output_dir>/<table_name>.csv` via
    DuckDB's native `COPY ... TO`, no pandas round-trip needed. Returns the
    list of table names actually exported."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    names = list(table_names) if table_names is not None else list_regularized_table_names(conn)
    for table_name in names:
        csv_path = str(output_dir / f"{table_name}.csv").replace("'", "''")
        conn.execute(f"COPY (SELECT * FROM \"{table_name}\") TO '{csv_path}' (HEADER, DELIMITER ',')")
    return names
