"""Stage implementation for parsing downloaded detail exports."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

from ..parsing.xls import normalize_subsection_label, parse_subsection_xls
from ..storage.database import (
    get_mine_ids_with_exports,
    get_stage_pending_mine_ids,
    mark_stage_complete,
    reset_stage_completion,
)
from ..storage.subsections import (
    SubsectionRecord,
    clear_stage_outputs,
    clear_subsection_stage_output,
    ensure_detail_tables,
    get_completed_stage_keys,
    insert_parsed_workbook,
    insert_parsed_cells,
    update_export_parse_metadata,
    upsert_subsection_stage_status,
)

logger = logging.getLogger(__name__)

_STAGE_NAME = "detail_parse"


def parse_detail_exports(
    conn,
    mine_ids: Iterable[str] | None = None,
    subsections: Iterable[str] | None = None,
    continue_on_error: bool = True,
    force: bool = False,
) -> dict[str, int]:
    """Parse downloaded subsection XLS exports into structured workbook rows."""
    subsection_filter = _normalize_subsection_filter(subsections)
    logger.info(
        "Starting detail parse (continue_on_error=%s, subsection_filter_count=%s, force=%s)",
        continue_on_error,
        len(subsection_filter) if subsection_filter is not None else "all",
        force,
    )
    ensure_detail_tables(conn)

    if mine_ids is None:
        pending_ids = get_stage_pending_mine_ids(conn, _STAGE_NAME) if not force else get_mine_ids_with_exports(conn)
    else:
        pending_ids = [str(mid) for mid in mine_ids]

    if force and pending_ids:
        reset_stage_completion(conn, pending_ids, _STAGE_NAME)
        for mine_id in pending_ids:
            clear_stage_outputs(conn, mine_id, _STAGE_NAME)

    export_rows = _load_export_rows(conn, pending_ids, subsection_filter=subsection_filter)
    logger.info("Loaded %d export row(s) for parsing.", len(export_rows))

    inserted_cells = 0
    parse_failures = 0
    parse_skips = 0

    rows_by_mine: dict[str, list[tuple[str, str, str, str, str | None]]] = {}
    for row in export_rows:
        rows_by_mine.setdefault(row[0], []).append(row)

    for mine_id, mine_rows in rows_by_mine.items():
        completed_keys = set() if force else get_completed_stage_keys(conn, mine_id, _STAGE_NAME)
        mine_had_failure = False

        for _, section_label, subsection_label, subsection_href, xls_path in mine_rows:
            key = (section_label, subsection_label)
            if key in completed_keys:
                parse_skips += 1
                continue

            path = Path(xls_path)
            record = SubsectionRecord(
                section_label=section_label,
                subsection_label=subsection_label,
                subsection_href=subsection_href or "",
            )
            if not path.exists():
                mine_had_failure = True
                parse_failures += 1
                upsert_subsection_stage_status(
                    conn,
                    mine_id,
                    record,
                    stage_name=_STAGE_NAME,
                    status="failed",
                    error_msg=f"Missing XLS file: {path}",
                )
                logger.warning("Missing XLS file for parse step: %s", path)
                continue

            clear_subsection_stage_output(conn, mine_id, record, _STAGE_NAME)
            try:
                parsed = parse_subsection_xls(path, subsection_label=subsection_label)
            except Exception as exc:
                mine_had_failure = True
                parse_failures += 1
                upsert_subsection_stage_status(
                    conn,
                    mine_id,
                    record,
                    stage_name=_STAGE_NAME,
                    status="failed",
                    error_msg=str(exc),
                )
                logger.warning(
                    "Failed to parse XLS (mine_id=%s, subsection=%s, path=%s): %s",
                    mine_id,
                    subsection_label,
                    path,
                    exc,
                )
                if continue_on_error:
                    continue
                raise

            update_export_parse_metadata(
                conn,
                mine_id,
                record,
                str(path),
                xls_sha256=parsed.xls_sha256,
                workbook_title=parsed.workbook_title,
                workbook_subtitle=parsed.workbook_subtitle,
                primary_sheet_name=parsed.primary_sheet_name,
                content_subsection_label=parsed.content_subsection_label,
            )
            insert_parsed_workbook(conn, mine_id, record, parsed)
            inserted_count = insert_parsed_cells(
                conn,
                mine_id,
                subsection_label,
                parsed.flat_cells,
            )
            inserted_cells += inserted_count
            upsert_subsection_stage_status(
                conn,
                mine_id,
                record,
                stage_name=_STAGE_NAME,
                status="completed",
            )
            logger.debug(
                "Inserted %d parsed cell row(s) for mine_id=%s subsection=%s",
                inserted_count,
                mine_id,
                subsection_label,
            )

        if not mine_had_failure and mine_rows:
            mark_stage_complete(conn, mine_id, _STAGE_NAME)

    logger.info(
        "Detail parse complete: %d export file(s), %d inserted compatibility cell row(s), %d skipped export(s), %d parse failure(s).",
        len(export_rows),
        inserted_cells,
        parse_skips,
        parse_failures,
    )
    return {
        "export_file_count": len(export_rows),
        "inserted_cell_count": inserted_cells,
        "skipped_export_count": parse_skips,
        "parse_failure_count": parse_failures,
    }


def _load_export_rows(
    conn,
    mine_ids: Iterable[str],
    subsection_filter: set[str] | None = None,
) -> list[tuple[str, str, str, str | None, str]]:
    mine_ids_list = [str(mid) for mid in mine_ids]
    if not mine_ids_list:
        return []
    placeholders = ",".join(["?"] * len(mine_ids_list))
    rows = conn.execute(
        f"""
        SELECT mine_id, section_label, subsection_label, subsection_href, xls_path
        FROM mine_subsection_exports
        WHERE mine_id IN ({placeholders})
        ORDER BY mine_id, section_label, subsection_label, xls_path
        """,
        mine_ids_list,
    ).fetchall()
    if subsection_filter is None:
        return rows
    return [
        row
        for row in rows
        if normalize_subsection_label(row[2]) in subsection_filter
    ]


def _normalize_subsection_filter(subsections: Iterable[str] | None) -> set[str] | None:
    if subsections is None:
        return None
    normalized = {
        normalize_subsection_label(str(subsection))
        for subsection in subsections
        if str(subsection).strip()
    }
    return normalized or None
