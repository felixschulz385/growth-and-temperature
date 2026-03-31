"""Shared helpers for subsection discovery, export metadata, and parsed cells."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Mapping

_PROPERTY_PROFILE_LABEL = "property profile"


@dataclass(frozen=True, slots=True)
class SubsectionRecord:
    section_label: str
    subsection_label: str
    subsection_href: str


def ensure_detail_tables(conn) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS mine_subsections (
            mine_id          TEXT NOT NULL,
            section_label    TEXT NOT NULL,
            subsection_label TEXT NOT NULL,
            subsection_href  TEXT,
            discovered_at    TIMESTAMPTZ NOT NULL,
            PRIMARY KEY (mine_id, section_label, subsection_label)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS mine_subsection_exports (
            mine_id          TEXT NOT NULL,
            section_label    TEXT NOT NULL,
            subsection_label TEXT NOT NULL,
            subsection_href  TEXT,
            xls_path         TEXT NOT NULL,
            xls_sha256       TEXT,
            workbook_title   TEXT,
            workbook_subtitle TEXT,
            primary_sheet_name TEXT,
            content_subsection_label TEXT,
            exported_at      TIMESTAMPTZ NOT NULL
        )
    """)
    conn.execute("ALTER TABLE mine_subsection_exports ADD COLUMN IF NOT EXISTS xls_sha256 TEXT")
    conn.execute("ALTER TABLE mine_subsection_exports ADD COLUMN IF NOT EXISTS workbook_title TEXT")
    conn.execute("ALTER TABLE mine_subsection_exports ADD COLUMN IF NOT EXISTS workbook_subtitle TEXT")
    conn.execute("ALTER TABLE mine_subsection_exports ADD COLUMN IF NOT EXISTS primary_sheet_name TEXT")
    conn.execute("ALTER TABLE mine_subsection_exports ADD COLUMN IF NOT EXISTS content_subsection_label TEXT")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS mine_subsection_cells (
            mine_id          TEXT NOT NULL,
            subsection_label TEXT NOT NULL,
            sheet_name       TEXT NOT NULL,
            row_index        INTEGER NOT NULL,
            column_name      TEXT NOT NULL,
            value            TEXT NOT NULL,
            parsed_at        TIMESTAMPTZ NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS mine_subsection_blocks (
            mine_id                 TEXT NOT NULL,
            section_label           TEXT NOT NULL,
            subsection_label        TEXT NOT NULL,
            xls_path                TEXT NOT NULL,
            xls_sha256              TEXT,
            sheet_name              TEXT NOT NULL,
            sheet_index             INTEGER NOT NULL,
            block_index             INTEGER NOT NULL,
            block_type              TEXT NOT NULL,
            block_title             TEXT,
            row_start               INTEGER NOT NULL,
            row_end                 INTEGER NOT NULL,
            header_row_count        INTEGER NOT NULL DEFAULT 0,
            workbook_title          TEXT,
            workbook_subtitle       TEXT,
            primary_sheet_name      TEXT,
            content_subsection_label TEXT,
            parsed_at               TIMESTAMPTZ NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS mine_subsection_block_cells (
            mine_id                 TEXT NOT NULL,
            section_label           TEXT NOT NULL,
            subsection_label        TEXT NOT NULL,
            xls_path                TEXT NOT NULL,
            xls_sha256              TEXT,
            sheet_name              TEXT NOT NULL,
            sheet_index             INTEGER NOT NULL,
            block_index             INTEGER NOT NULL,
            block_type              TEXT NOT NULL,
            block_title             TEXT,
            row_number              INTEGER NOT NULL,
            column_index            INTEGER NOT NULL,
            column_name             TEXT NOT NULL,
            cell_ref                TEXT NOT NULL,
            cell_role               TEXT NOT NULL,
            value                   TEXT NOT NULL,
            parsed_at               TIMESTAMPTZ NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS mine_property_geometries (
            mine_id          TEXT NOT NULL,
            geometry_kind    TEXT NOT NULL,
            geometry_wkt     TEXT NOT NULL,
            bounds_minx      DOUBLE,
            bounds_miny      DOUBLE,
            bounds_maxx      DOUBLE,
            bounds_maxy      DOUBLE,
            extracted_at     TIMESTAMPTZ NOT NULL,
            PRIMARY KEY (mine_id, geometry_kind)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS mine_subsection_stage_status (
            mine_id          TEXT NOT NULL,
            section_label    TEXT NOT NULL,
            subsection_label TEXT NOT NULL,
            stage_name       TEXT NOT NULL,
            subsection_href  TEXT,
            status           TEXT NOT NULL,
            last_attempted_at TIMESTAMPTZ NOT NULL,
            completed_at     TIMESTAMPTZ,
            error_msg        TEXT,
            PRIMARY KEY (mine_id, section_label, subsection_label, stage_name)
        )
    """)


def build_subsection_records(
    subsections: Mapping[str, Iterable[Mapping[str, str]]],
) -> list[SubsectionRecord]:
    records: list[SubsectionRecord] = []
    for section_label, items in subsections.items():
        for item in items:
            subsection_label = item.get("label", "").strip()
            if not subsection_label:
                continue
            records.append(
                SubsectionRecord(
                    section_label=section_label.strip(),
                    subsection_label=subsection_label,
                    subsection_href=(item.get("href") or "").strip(),
                )
            )
    return sorted(
        records,
        key=lambda record: (
            record.section_label.casefold(),
            record.subsection_label.casefold(),
            record.subsection_href,
        ),
    )


def clear_detail_rows(conn, mine_id: str, include_cells: bool) -> None:
    conn.execute("DELETE FROM mine_subsections WHERE mine_id = ?", [mine_id])
    conn.execute("DELETE FROM mine_subsection_exports WHERE mine_id = ?", [mine_id])
    conn.execute("DELETE FROM mine_property_geometries WHERE mine_id = ?", [mine_id])
    conn.execute("DELETE FROM mine_subsection_stage_status WHERE mine_id = ?", [mine_id])
    if include_cells:
        conn.execute("DELETE FROM mine_subsection_cells WHERE mine_id = ?", [mine_id])
        conn.execute("DELETE FROM mine_subsection_blocks WHERE mine_id = ?", [mine_id])
        conn.execute("DELETE FROM mine_subsection_block_cells WHERE mine_id = ?", [mine_id])


def clear_stage_outputs(conn, mine_id: str, stage_name: str) -> None:
    if stage_name == "detail_exports":
        conn.execute("DELETE FROM mine_subsection_exports WHERE mine_id = ?", [mine_id])
        conn.execute("DELETE FROM mine_property_geometries WHERE mine_id = ?", [mine_id])
        conn.execute(
            "DELETE FROM mine_subsection_stage_status WHERE mine_id = ? AND stage_name IN ('detail_exports', 'detail_parse')",
            [mine_id],
        )
        conn.execute("DELETE FROM mine_subsection_cells WHERE mine_id = ?", [mine_id])
        conn.execute("DELETE FROM mine_subsection_blocks WHERE mine_id = ?", [mine_id])
        conn.execute("DELETE FROM mine_subsection_block_cells WHERE mine_id = ?", [mine_id])
        return
    if stage_name == "detail_parse":
        conn.execute("DELETE FROM mine_subsection_cells WHERE mine_id = ?", [mine_id])
        conn.execute("DELETE FROM mine_subsection_blocks WHERE mine_id = ?", [mine_id])
        conn.execute("DELETE FROM mine_subsection_block_cells WHERE mine_id = ?", [mine_id])
        conn.execute(
            "DELETE FROM mine_subsection_stage_status WHERE mine_id = ? AND stage_name = 'detail_parse'",
            [mine_id],
        )
        return
    raise ValueError(f"Unsupported stage name: {stage_name}")


def clear_subsection_stage_output(
    conn,
    mine_id: str,
    record: SubsectionRecord,
    stage_name: str,
) -> None:
    params = [mine_id, record.section_label, record.subsection_label]
    if stage_name == "detail_exports":
        conn.execute(
            """
            DELETE FROM mine_subsection_exports
            WHERE mine_id = ? AND section_label = ? AND subsection_label = ?
            """,
            params,
        )
        if record.subsection_label.strip().casefold() == _PROPERTY_PROFILE_LABEL:
            conn.execute("DELETE FROM mine_property_geometries WHERE mine_id = ?", [mine_id])
        conn.execute(
            """
            DELETE FROM mine_subsection_stage_status
            WHERE mine_id = ? AND section_label = ? AND subsection_label = ?
              AND stage_name IN ('detail_exports', 'detail_parse')
            """,
            params,
        )
        conn.execute(
            """
            DELETE FROM mine_subsection_cells
            WHERE mine_id = ? AND subsection_label = ?
            """,
            [mine_id, record.subsection_label],
        )
        conn.execute(
            """
            DELETE FROM mine_subsection_blocks
            WHERE mine_id = ? AND section_label = ? AND subsection_label = ?
            """,
            params,
        )
        conn.execute(
            """
            DELETE FROM mine_subsection_block_cells
            WHERE mine_id = ? AND section_label = ? AND subsection_label = ?
            """,
            params,
        )
        return
    if stage_name == "detail_parse":
        conn.execute(
            """
            DELETE FROM mine_subsection_cells
            WHERE mine_id = ? AND subsection_label = ?
            """,
            [mine_id, record.subsection_label],
        )
        conn.execute(
            """
            DELETE FROM mine_subsection_blocks
            WHERE mine_id = ? AND section_label = ? AND subsection_label = ?
            """,
            params,
        )
        conn.execute(
            """
            DELETE FROM mine_subsection_block_cells
            WHERE mine_id = ? AND section_label = ? AND subsection_label = ?
            """,
            params,
        )
        conn.execute(
            """
            DELETE FROM mine_subsection_stage_status
            WHERE mine_id = ? AND section_label = ? AND subsection_label = ?
              AND stage_name = 'detail_parse'
            """,
            params,
        )
        return
    raise ValueError(f"Unsupported stage name: {stage_name}")


def insert_subsection_records(conn, mine_id: str, records: Iterable[SubsectionRecord]) -> int:
    rows = [
        (mine_id, record.section_label, record.subsection_label, record.subsection_href)
        for record in records
    ]
    if not rows:
        return 0
    conn.executemany(
        """
        INSERT INTO mine_subsections (
            mine_id, section_label, subsection_label, subsection_href, discovered_at
        ) VALUES (?, ?, ?, ?, now())
        ON CONFLICT (mine_id, section_label, subsection_label)
        DO UPDATE SET
            subsection_href = excluded.subsection_href,
            discovered_at = excluded.discovered_at
        """,
        rows,
    )
    return len(rows)


def insert_geometry_rows(conn, mine_id: str, geometries: Mapping[str, object]) -> int:
    rows = [
        (
            mine_id,
            geometry_kind,
            geometry.wkt,
            float(geometry.bounds[0]),
            float(geometry.bounds[1]),
            float(geometry.bounds[2]),
            float(geometry.bounds[3]),
        )
        for geometry_kind, geometry in geometries.items()
    ]
    if not rows:
        return 0
    conn.executemany(
        """
        INSERT INTO mine_property_geometries (
            mine_id, geometry_kind, geometry_wkt,
            bounds_minx, bounds_miny, bounds_maxx, bounds_maxy,
            extracted_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, now())
        """,
        rows,
    )
    return len(rows)


def insert_export_row(conn, mine_id: str, record: SubsectionRecord, xls_path: str) -> None:
    conn.execute(
        """
        INSERT INTO mine_subsection_exports (
            mine_id, section_label, subsection_label, subsection_href, xls_path, exported_at
        ) VALUES (?, ?, ?, ?, ?, now())
        """,
        [
            mine_id,
            record.section_label,
            record.subsection_label,
            record.subsection_href,
            xls_path,
        ],
    )


def update_export_parse_metadata(
    conn,
    mine_id: str,
    record: SubsectionRecord,
    xls_path: str,
    *,
    xls_sha256: str | None,
    workbook_title: str | None,
    workbook_subtitle: str | None,
    primary_sheet_name: str | None,
    content_subsection_label: str | None,
) -> None:
    conn.execute(
        """
        UPDATE mine_subsection_exports
        SET xls_sha256 = ?,
            workbook_title = ?,
            workbook_subtitle = ?,
            primary_sheet_name = ?,
            content_subsection_label = ?
        WHERE mine_id = ?
          AND section_label = ?
          AND subsection_label = ?
          AND xls_path = ?
        """,
        [
            xls_sha256,
            workbook_title,
            workbook_subtitle,
            primary_sheet_name,
            content_subsection_label,
            mine_id,
            record.section_label,
            record.subsection_label,
            xls_path,
        ],
    )


def insert_parsed_cells(
    conn,
    mine_id: str,
    subsection_label: str,
    parsed_cells: Iterable[object],
) -> int:
    rows = [
        (
            mine_id,
            subsection_label,
            cell.sheet_name,
            cell.row_index,
            cell.column_name,
            cell.value,
        )
        for cell in parsed_cells
    ]
    if not rows:
        return 0
    conn.executemany(
        """
        INSERT INTO mine_subsection_cells (
            mine_id, subsection_label, sheet_name, row_index, column_name, value, parsed_at
        ) VALUES (?, ?, ?, ?, ?, ?, now())
        """,
        rows,
    )
    return len(rows)


def insert_parsed_workbook(
    conn,
    mine_id: str,
    record: SubsectionRecord,
    parsed_workbook,
) -> tuple[int, int]:
    block_rows = []
    cell_rows = []
    parsed_at = datetime.now(timezone.utc)

    for sheet in parsed_workbook.sheets:
        for block in sheet.blocks:
            block_rows.append(
                (
                    mine_id,
                    record.section_label,
                    record.subsection_label,
                    str(parsed_workbook.xls_path),
                    parsed_workbook.xls_sha256,
                    sheet.sheet_name,
                    sheet.sheet_index,
                    block.block_index,
                    block.block_type,
                    block.block_title,
                    block.row_start,
                    block.row_end,
                    block.header_row_count,
                    parsed_workbook.workbook_title,
                    parsed_workbook.workbook_subtitle,
                    parsed_workbook.primary_sheet_name,
                    parsed_workbook.content_subsection_label,
                    parsed_at,
                )
            )
            for cell in block.cells:
                cell_rows.append(
                    (
                        mine_id,
                        record.section_label,
                        record.subsection_label,
                        str(parsed_workbook.xls_path),
                        parsed_workbook.xls_sha256,
                        sheet.sheet_name,
                        sheet.sheet_index,
                        block.block_index,
                        block.block_type,
                        block.block_title,
                        cell.row_number,
                        cell.column_index,
                        cell.column_name,
                        cell.cell_ref,
                        cell.role,
                        cell.value,
                        parsed_at,
                    )
                )

    if block_rows:
        conn.executemany(
            """
            INSERT INTO mine_subsection_blocks (
                mine_id,
                section_label,
                subsection_label,
                xls_path,
                xls_sha256,
                sheet_name,
                sheet_index,
                block_index,
                block_type,
                block_title,
                row_start,
                row_end,
                header_row_count,
                workbook_title,
                workbook_subtitle,
                primary_sheet_name,
                content_subsection_label,
                parsed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            block_rows,
        )
    if cell_rows:
        conn.executemany(
            """
            INSERT INTO mine_subsection_block_cells (
                mine_id,
                section_label,
                subsection_label,
                xls_path,
                xls_sha256,
                sheet_name,
                sheet_index,
                block_index,
                block_type,
                block_title,
                row_number,
                column_index,
                column_name,
                cell_ref,
                cell_role,
                value,
                parsed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            cell_rows,
        )
    return len(block_rows), len(cell_rows)


def get_completed_stage_keys(conn, mine_id: str, stage_name: str) -> set[tuple[str, str]]:
    rows = conn.execute(
        """
        SELECT section_label, subsection_label
        FROM mine_subsection_stage_status
        WHERE mine_id = ?
          AND stage_name = ?
          AND status = 'completed'
          AND completed_at IS NOT NULL
        """,
        [mine_id, stage_name],
    ).fetchall()
    return {(row[0], row[1]) for row in rows}


def upsert_subsection_stage_status(
    conn,
    mine_id: str,
    record: SubsectionRecord,
    stage_name: str,
    status: str,
    error_msg: str | None = None,
) -> None:
    now = datetime.now(timezone.utc)
    completed_at = now if status == "completed" else None
    conn.execute(
        """
        INSERT INTO mine_subsection_stage_status (
            mine_id,
            section_label,
            subsection_label,
            stage_name,
            subsection_href,
            status,
            last_attempted_at,
            completed_at,
            error_msg
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (mine_id, section_label, subsection_label, stage_name)
        DO UPDATE SET
            subsection_href = excluded.subsection_href,
            status = excluded.status,
            last_attempted_at = excluded.last_attempted_at,
            completed_at = excluded.completed_at,
            error_msg = excluded.error_msg
        """,
        [
            mine_id,
            record.section_label,
            record.subsection_label,
            stage_name,
            record.subsection_href,
            status,
            now,
            completed_at,
            error_msg,
        ],
    )
