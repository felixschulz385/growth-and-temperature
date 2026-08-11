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
from pathlib import Path
from typing import Iterable

from ..parsing.xls import normalize_subsection_label, parse_subsection_xls
from ..regularize.registry import STATUS_COMPLETED, STATUS_UNVERIFIED, classify_and_regularize
from ..storage.database import (
    get_mine_ids_with_exports,
    get_stage_pending_mine_ids,
    mark_stage_complete,
    reset_stage_completion,
)
from ..storage.regularized import persist_regularized_tables
from ..storage.subsections import (
    SubsectionRecord,
    get_completed_stage_keys,
    upsert_subsection_stage_status,
)

logger = logging.getLogger(__name__)

_STAGE_NAME = "detail_regularize"


def regularize_detail_exports(
    conn,
    mine_ids: Iterable[str] | None = None,
    subsections: Iterable[str] | None = None,
    continue_on_error: bool = True,
    force: bool = False,
) -> dict[str, int]:
    """Regularize downloaded subsection XLS exports into typed, per-type
    DuckDB tables (see `regularize/registry.py` for the fixed subsection-type
    list)."""
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
        "content_mismatch_count": 0,
        "unverified_count": 0,
        "unknown_type_count": 0,
        "failed_count": 0,
        "skipped_count": 0,
        "regularized_row_count": 0,
    }

    rows_by_mine: dict[str, list[tuple]] = {}
    for row in export_rows:
        rows_by_mine.setdefault(row[0], []).append(row)

    for mine_id, mine_rows in rows_by_mine.items():
        completed_keys = set() if force else get_completed_stage_keys(conn, mine_id, _STAGE_NAME)
        mine_had_failure = False

        for _, section_label, subsection_label, subsection_href, xls_path, xls_sha256 in mine_rows:
            key = (section_label, subsection_label)
            if key in completed_keys:
                counts["skipped_count"] += 1
                continue

            record = SubsectionRecord(
                section_label=section_label,
                subsection_label=subsection_label,
                subsection_href=subsection_href or "",
            )
            path = Path(xls_path)
            if not path.exists():
                mine_had_failure = True
                counts["failed_count"] += 1
                upsert_subsection_stage_status(
                    conn, mine_id, record, stage_name=_STAGE_NAME, status="failed",
                    error_msg=f"Missing XLS file: {path}",
                )
                logger.warning("Missing XLS file for regularize step: %s", path)
                continue

            try:
                parsed = parse_subsection_xls(path, subsection_label=subsection_label)
                status, tables = classify_and_regularize(parsed, mine_id, subsection_label)
            except Exception as exc:
                mine_had_failure = True
                counts["failed_count"] += 1
                upsert_subsection_stage_status(
                    conn, mine_id, record, stage_name=_STAGE_NAME, status="failed", error_msg=str(exc),
                )
                logger.warning(
                    "Failed to regularize XLS (mine_id=%s, subsection=%s, path=%s): %s",
                    mine_id, subsection_label, path, exc,
                )
                if continue_on_error:
                    continue
                raise

            if status in (STATUS_COMPLETED, STATUS_UNVERIFIED) and tables:
                row_counts = persist_regularized_tables(conn, mine_id, parsed.xls_sha256 or xls_sha256, tables)
                counts["regularized_row_count"] += sum(row_counts.values())

            counts[f"{status}_count"] = counts.get(f"{status}_count", 0) + 1
            upsert_subsection_stage_status(conn, mine_id, record, stage_name=_STAGE_NAME, status=status)
            logger.debug(
                "Regularized mine_id=%s subsection=%s -> status=%s",
                mine_id, subsection_label, status,
            )

        if not mine_had_failure and mine_rows:
            mark_stage_complete(conn, mine_id, _STAGE_NAME)

    logger.info(
        "Detail regularize complete: %d export file(s), %d completed, %d content_mismatch, "
        "%d unverified, %d unknown_type, %d failed, %d skipped, %d row(s) written.",
        counts["export_file_count"], counts["completed_count"], counts["content_mismatch_count"],
        counts["unverified_count"], counts["unknown_type_count"], counts["failed_count"],
        counts["skipped_count"], counts["regularized_row_count"],
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
