"""Stage implementation for regularizing parsed detail exports into typed,
per-subsection-type DuckDB tables. Mirrors `parse_detail_exports.py`'s
structure (pending-mine resolution, force/continue_on_error handling,
per-(mine, section, subsection) stage-status tracking), operating one layer
downstream: where `detail_parse` turns raw xlsx into generic block/cell
rows, this turns those same xlsx exports into the fixed, typed
`detail_*` tables documented in `regularize/registry.py`.
"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor
from typing import Iterable, Iterator

from tqdm import tqdm

from ..parsing.xls import normalize_subsection_label
from ..regularize.registry import STATUS_COMPLETED, STATUS_RECLASSIFIED, STATUS_UNVERIFIED
from ..storage.database import (
    get_mine_ids_with_exports,
    get_stage_pending_mine_ids,
    mark_stage_complete,
    reset_stage_completion,
)
from ..storage.regularized import persist_regularized_tables
from ..storage.subsections import get_completed_stage_keys, upsert_subsection_stage_status
from ._regularize_worker import ExportRow as _ExportRow
from ._regularize_worker import RowResult as _RowResult
from ._regularize_worker import process_one_row as _process_one_row

logger = logging.getLogger(__name__)

_STAGE_NAME = "detail_regularize"
_DEFAULT_MAX_WORKERS = 8
#: `ProcessPoolExecutor.map`'s default chunksize=1 means one IPC round-trip
#: (pickle args, enqueue, worker unpickles+runs+pickles result, dequeue) per
#: row -- fine for a handful of rows, but at real scale (hundreds of
#: thousands) that per-task overhead dominates over the tiny amount of
#: actual work each row does. Batching several rows per IPC round-trip
#: amortizes that overhead; `concurrent.futures`' own docs recommend a
#: larger chunksize for exactly this "many small, fast tasks" shape.
_CHUNKSIZE_DIVISOR = 4
#: Cap on the computed chunksize -- a chunk's future only resolves once every
#: row in it is done, so an unbounded chunk both makes the progress bar jump
#: in large bursts and means one pathologically slow file inside a chunk
#: delays every other row in that same chunk from being reported/persisted.
_MAX_CHUNKSIZE = 200


def _consume_results(
    results: Iterator[_RowResult],
    work_items: list[_ExportRow],
    conn,
    counts: dict[str, int],
    mine_had_failure: dict[str, bool],
    *,
    continue_on_error: bool,
) -> None:
    """Consume worker results in submission order on the main process,
    doing every DuckDB write serially (see module docstring)."""
    for result in tqdm(results, total=len(work_items), desc="detail_regularize", unit="file"):
        mine_id, record = result.mine_id, result.record
        if result.error is not None:
            mine_had_failure[mine_id] = True
            counts["failed_count"] += 1
            upsert_subsection_stage_status(
                conn, mine_id, record, stage_name=_STAGE_NAME, status="failed",
                error_msg=result.error,
            )
            logger.warning(
                "Failed to regularize XLS (mine_id=%s, subsection=%s): %s",
                mine_id, record.subsection_label, result.error,
            )
            if not continue_on_error:
                raise RuntimeError(
                    f"Failed to regularize XLS (mine_id={mine_id}, "
                    f"subsection={record.subsection_label}): {result.error}"
                )
            continue

        status = result.status
        if status in (STATUS_COMPLETED, STATUS_RECLASSIFIED, STATUS_UNVERIFIED) and result.tables:
            row_counts = persist_regularized_tables(conn, mine_id, result.xls_sha256, result.tables)
            counts["regularized_row_count"] += sum(row_counts.values())

        counts[f"{status}_count"] = counts.get(f"{status}_count", 0) + 1
        upsert_subsection_stage_status(conn, mine_id, record, stage_name=_STAGE_NAME, status=status)
        logger.debug(
            "Regularized mine_id=%s subsection=%s -> status=%s",
            mine_id, record.subsection_label, status,
        )


def regularize_detail_exports(
    conn,
    mine_ids: Iterable[str] | None = None,
    subsections: Iterable[str] | None = None,
    continue_on_error: bool = True,
    force: bool = False,
    max_workers: int = _DEFAULT_MAX_WORKERS,
) -> dict[str, int]:
    """Regularize downloaded subsection XLS exports into typed, per-type
    DuckDB tables (see `regularize/registry.py` for the fixed subsection-type
    list).

    Parsing + classification (`_regularize_worker.process_one_row`, no
    DuckDB access) runs across a `ProcessPoolExecutor` of *max_workers*
    processes when *max_workers* > 1. This is a deliberate departure from
    `src/data/common/hpc/push.py`'s `push_units_concurrent()` precedent
    (threads for blocking I/O like rsync/ssh) -- `parsing/xls.py` parses each
    workbook's XML by hand (`zipfile` + `xml.etree.ElementTree`, building one
    dataclass per cell), which is pure-Python CPU work, not blocking I/O.
    Confirmed empirically on a real 332k-row run: `ThreadPoolExecutor` gave
    no speedup (all worker threads contend for the single GIL on that
    Python-level parsing work). `max_workers <= 1` skips the pool entirely
    and calls the worker function directly in-process -- used by tests,
    since a monkeypatched `parse_subsection_xls`/`classify_and_regularize`
    is invisible to a spawned worker process (it re-imports the target
    module fresh).

    Two things that matter at real scale (hundreds of thousands of rows),
    both found by actually running this against the real local scraper DB
    and watching it stall at 0% with worker processes started but no
    progress:

    - The worker function lives in `_regularize_worker.py`, not here,
      specifically to keep each spawned worker's own import graph
      stdlib-only. Importing *any* `src.data.sources.snl_mining.*`
      submodule first runs the package's `__init__.py` -- which used to
      eagerly `from .source import SnlMiningSource`, dragging in that
      module's full pipeline stack (`geobox` -> `pandas` -> `pyarrow`) into
      every one of `max_workers` freshly-spawned processes just to run a
      few hundred bytes of XML parsing. The package `__init__.py` is now
      lazy (PEP 562) for exactly this reason.
    - `pool.map(..., chunksize=1)` (the default) means one full IPC
      round-trip per row; at 332k rows that overhead dominates the actual
      (tiny) per-row work. `chunksize` below batches rows per round-trip.

    All DuckDB writes (persisting regularized tables, stage-status upserts,
    `mark_stage_complete`) stay serialized on the main process/thread as
    results stream back, since a single `DuckDBPyConnection` isn't safe for
    concurrent use and never crosses the process boundary either way.
    """
    subsection_filter = _normalize_subsection_filter(subsections)
    logger.info(
        "Starting detail regularize (continue_on_error=%s, subsection_filter_count=%s, force=%s)",
        continue_on_error,
        len(subsection_filter) if subsection_filter is not None else "all",
        force,
    )

    if mine_ids is None:
        pending_ids = get_stage_pending_mine_ids(conn, _STAGE_NAME) if not force else get_mine_ids_with_exports(conn)
    else:
        pending_ids = [str(mid) for mid in mine_ids]

    if force and pending_ids:
        reset_stage_completion(conn, pending_ids, _STAGE_NAME)
        placeholders = ",".join(["?"] * len(pending_ids))
        conn.execute(
            f"""
            DELETE FROM mine_subsection_stage_status
            WHERE mine_id IN ({placeholders}) AND stage_name = ?
            """,
            [*pending_ids, _STAGE_NAME],
        )

    export_rows = _load_export_rows(conn, pending_ids, subsection_filter=subsection_filter)
    logger.info("Loaded %d export row(s) for regularization.", len(export_rows))

    counts = {
        "export_file_count": len(export_rows),
        "completed_count": 0,
        "reclassified_count": 0,
        "content_mismatch_count": 0,
        "unverified_count": 0,
        "unknown_type_count": 0,
        "failed_count": 0,
        "skipped_count": 0,
        "regularized_row_count": 0,
    }

    rows_by_mine: dict[str, list[_ExportRow]] = {}
    for row in export_rows:
        rows_by_mine.setdefault(row[0], []).append(row)

    # Per-mine "did anything fail" tracking, decoupled from processing order
    # now that rows are dispatched to a worker pool rather than iterated
    # mine-by-mine -- mark_stage_complete still needs one bool per mine.
    mine_had_failure: dict[str, bool] = {mine_id: False for mine_id in rows_by_mine}

    work_items: list[_ExportRow] = []
    for mine_id, mine_rows in rows_by_mine.items():
        completed_keys = set() if force else get_completed_stage_keys(conn, mine_id, _STAGE_NAME)
        for row in mine_rows:
            _, section_label, subsection_label, *_rest = row
            if (section_label, subsection_label) in completed_keys:
                counts["skipped_count"] += 1
                continue
            work_items.append(row)

    if work_items:
        if max_workers <= 1:
            results: Iterator[_RowResult] = map(_process_one_row, work_items)
            _consume_results(
                results, work_items, conn, counts, mine_had_failure, continue_on_error=continue_on_error,
            )
        else:
            # Parse+classify (CPU-bound, no DuckDB access) runs across
            # `max_workers` processes; results are consumed here, on the
            # main process, in submission order -- every DuckDB write stays
            # serial (module docstring: a single connection isn't safe for
            # concurrent use, and never crosses the process boundary).
            chunksize = min(_MAX_CHUNKSIZE, max(1, len(work_items) // (max_workers * _CHUNKSIZE_DIVISOR)))
            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                _consume_results(
                    pool.map(_process_one_row, work_items, chunksize=chunksize),
                    work_items, conn, counts, mine_had_failure, continue_on_error=continue_on_error,
                )

    for mine_id, mine_rows in rows_by_mine.items():
        if not mine_had_failure[mine_id] and mine_rows:
            mark_stage_complete(conn, mine_id, _STAGE_NAME)

    logger.info(
        "Detail regularize complete: %d export file(s), %d completed, %d reclassified, "
        "%d content_mismatch, %d unverified, %d unknown_type, %d failed, %d skipped, %d row(s) written.",
        counts["export_file_count"], counts["completed_count"], counts["reclassified_count"],
        counts["content_mismatch_count"], counts["unverified_count"], counts["unknown_type_count"],
        counts["failed_count"], counts["skipped_count"], counts["regularized_row_count"],
    )
    return counts


def _load_export_rows(
    conn,
    mine_ids: Iterable[str],
    subsection_filter: set[str] | None = None,
) -> list[tuple[str, str, str, str | None, str, str | None]]:
    mine_ids_list = [str(mid) for mid in mine_ids]
    if not mine_ids_list:
        return []
    placeholders = ",".join(["?"] * len(mine_ids_list))
    rows = conn.execute(
        f"""
        SELECT mine_id, section_label, subsection_label, subsection_href, xls_path, xls_sha256
        FROM mine_subsection_exports
        WHERE mine_id IN ({placeholders})
        ORDER BY mine_id, section_label, subsection_label, xls_path
        """,
        mine_ids_list,
    ).fetchall()
    if subsection_filter is None:
        return rows
    return [row for row in rows if normalize_subsection_label(row[2]) in subsection_filter]


def _normalize_subsection_filter(subsections: Iterable[str] | None) -> set[str] | None:
    if subsections is None:
        return None
    normalized = {
        normalize_subsection_label(str(subsection))
        for subsection in subsections
        if str(subsection).strip()
    }
    return normalized or None
