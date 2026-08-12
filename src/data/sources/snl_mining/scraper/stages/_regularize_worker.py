"""The per-row parse+classify work `regularize_detail_exports.py` dispatches
to a `ProcessPoolExecutor` -- deliberately split into its own module with a
minimal import graph.

Every worker process re-imports whatever module `_process_one_row` lives in
from scratch on startup (`ProcessPoolExecutor` pickles by module+qualname,
not by value). Importing any `src.data.sources.snl_mining.*` submodule
first runs the package's own `__init__.py` -- which is lazy (see
`snl_mining/__init__.py`) specifically so that doesn't drag in `.source`'s
full pipeline stack (geobox -> pandas -> pyarrow) here. Keeping this
module's own transitive imports to stdlib-only (`parsing/xls.py`,
`regularize/registry.py`, and `storage/subsections.py`'s `SubsectionRecord`
are all stdlib-only -- no pandas/duckdb) means each of the worker processes
`regularize_detail_exports.py` spawns starts fast, rather than each
re-importing pandas/pyarrow/duckdb for functionality (`persist_regularized_tables`,
`get_connection`, ...) only the main process actually uses.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..parsing.xls import parse_subsection_xls
from ..regularize.registry import classify_and_regularize
from ..storage.subsections import SubsectionRecord

#: One row from `mine_subsection_exports`, as loaded by `_load_export_rows`.
ExportRow = tuple[str, str, str, str | None, str, str | None]


@dataclass(frozen=True, slots=True)
class RowResult:
    """Result of parsing+classifying one export row -- no `conn` access, so
    this is what worker processes produce; every field needed to persist the
    result back on the main process is carried here rather than re-derived
    (and must stay picklable, to cross the process boundary)."""

    mine_id: str
    record: SubsectionRecord
    xls_sha256: str | None
    status: str | None
    tables: dict[str, list[dict]]
    error: str | None


def process_one_row(row: ExportRow) -> RowResult:
    """Parse + classify one export row. Runs in a worker process -- must not
    touch a DuckDB connection, which never crosses the process boundary."""
    mine_id, section_label, subsection_label, subsection_href, xls_path, xls_sha256 = row
    record = SubsectionRecord(
        section_label=section_label,
        subsection_label=subsection_label,
        subsection_href=subsection_href or "",
    )
    path = Path(xls_path)
    if not path.exists():
        return RowResult(mine_id, record, xls_sha256, None, {}, f"Missing XLS file: {path}")

    try:
        parsed = parse_subsection_xls(path, subsection_label=subsection_label)
        status, tables = classify_and_regularize(parsed, mine_id, subsection_label)
    except Exception as exc:  # noqa: BLE001 -- surfaced as a per-row failure, not raised in-worker
        return RowResult(mine_id, record, xls_sha256, None, {}, str(exc))

    return RowResult(mine_id, record, parsed.xls_sha256 or xls_sha256, status, tables, None)
