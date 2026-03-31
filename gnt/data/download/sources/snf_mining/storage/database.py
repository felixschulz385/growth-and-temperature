"""DuckDB persistence layer for the SNF Mining scraper."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import duckdb

from ..config import DATA_DIR, DEFAULT_DB_PATH

logger = logging.getLogger(__name__)


def get_connection(db_path: str | Path = DEFAULT_DB_PATH) -> duckdb.DuckDBPyConnection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(db_path))
    _bootstrap(conn)
    logger.debug("DuckDB connection opened: %s", db_path)
    return conn


def _bootstrap(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS mines (
            mine_id            TEXT PRIMARY KEY,
            id_scraped_at      TIMESTAMPTZ NOT NULL,
            detail_scraped_at  TIMESTAMPTZ,
            detail_exports_completed_at TIMESTAMPTZ,
            detail_parse_completed_at   TIMESTAMPTZ
        )
    """)
    conn.execute("ALTER TABLE mines ADD COLUMN IF NOT EXISTS detail_scraped_at TIMESTAMPTZ")
    conn.execute("ALTER TABLE mines ADD COLUMN IF NOT EXISTS detail_exports_completed_at TIMESTAMPTZ")
    conn.execute("ALTER TABLE mines ADD COLUMN IF NOT EXISTS detail_parse_completed_at TIMESTAMPTZ")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS screener_state (
            screener_key   TEXT PRIMARY KEY,
            total_pages    INTEGER     NOT NULL,
            last_page_done INTEGER     NOT NULL DEFAULT 0,
            started_at     TIMESTAMPTZ NOT NULL,
            completed_at   TIMESTAMPTZ
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS scrape_errors (
            mine_id     TEXT,
            block_name  TEXT,
            error_msg   TEXT,
            occurred_at TIMESTAMPTZ NOT NULL
        )
    """)


def upsert_mine_ids(conn: duckdb.DuckDBPyConnection, mine_ids: Iterable[str]) -> int:
    now = datetime.now(timezone.utc)
    inserted = 0
    for mid in mine_ids:
        res = conn.execute("""
            INSERT OR IGNORE INTO mines (mine_id, id_scraped_at)
            VALUES (?, ?)
        """, [str(mid), now])
        if getattr(res, "rowcount", 0) == 1:
            inserted += 1
    logger.debug("Upserted %d new mine IDs.", inserted)
    return inserted


def get_unscraped_ids(conn: duckdb.DuckDBPyConnection) -> list[str]:
    rows = conn.execute("""
        SELECT mine_id FROM mines
        WHERE COALESCE(detail_exports_completed_at, detail_scraped_at) IS NULL
        ORDER BY id_scraped_at
    """).fetchall()
    return [r[0] for r in rows]


def mark_detail_scraped(conn: duckdb.DuckDBPyConnection, mine_id: str) -> None:
    now = datetime.now(timezone.utc)
    conn.execute("""
        UPDATE mines
        SET detail_scraped_at = ?,
            detail_exports_completed_at = COALESCE(detail_exports_completed_at, ?)
        WHERE mine_id = ?
    """, [now, now, mine_id])


def get_all_mine_ids(conn: duckdb.DuckDBPyConnection) -> list[str]:
    rows = conn.execute("""
        SELECT mine_id
        FROM mines
        ORDER BY id_scraped_at, mine_id
    """).fetchall()
    return [row[0] for row in rows]


def get_stage_pending_mine_ids(conn: duckdb.DuckDBPyConnection, stage_name: str) -> list[str]:
    if stage_name == "ids":
        return []
    if stage_name == "detail_exports":
        rows = conn.execute("""
            SELECT mine_id
            FROM mines
            WHERE detail_exports_completed_at IS NULL
            ORDER BY id_scraped_at, mine_id
        """).fetchall()
        return [row[0] for row in rows]
    if stage_name == "detail_parse":
        rows = conn.execute("""
            SELECT DISTINCT m.mine_id
            FROM mines AS m
            JOIN mine_subsection_exports AS e
              ON e.mine_id = m.mine_id
            WHERE m.detail_parse_completed_at IS NULL
            ORDER BY m.id_scraped_at, m.mine_id
        """).fetchall()
        return [row[0] for row in rows]
    raise ValueError(f"Unsupported stage name: {stage_name}")


def get_mine_ids_with_exports(conn: duckdb.DuckDBPyConnection) -> list[str]:
    rows = conn.execute("""
        SELECT DISTINCT m.mine_id
        FROM mines AS m
        JOIN mine_subsection_exports AS e
          ON e.mine_id = m.mine_id
        ORDER BY m.id_scraped_at, m.mine_id
    """).fetchall()
    return [row[0] for row in rows]


def mark_stage_complete(conn: duckdb.DuckDBPyConnection, mine_id: str, stage_name: str) -> None:
    now = datetime.now(timezone.utc)
    if stage_name == "detail_exports":
        conn.execute(
            """
            UPDATE mines
            SET detail_scraped_at = ?,
                detail_exports_completed_at = ?
            WHERE mine_id = ?
            """,
            [now, now, mine_id],
        )
        return
    if stage_name == "detail_parse":
        conn.execute(
            """
            UPDATE mines
            SET detail_parse_completed_at = ?
            WHERE mine_id = ?
            """,
            [now, mine_id],
        )
        return
    raise ValueError(f"Unsupported stage name: {stage_name}")


def reset_stage_completion(conn: duckdb.DuckDBPyConnection, mine_ids: Iterable[str], stage_name: str) -> None:
    mine_ids_list = [str(mine_id) for mine_id in mine_ids]
    if not mine_ids_list:
        return
    placeholders = ",".join(["?"] * len(mine_ids_list))
    if stage_name == "detail_exports":
        conn.execute(
            f"""
            UPDATE mines
            SET detail_scraped_at = NULL,
                detail_exports_completed_at = NULL,
                detail_parse_completed_at = NULL
            WHERE mine_id IN ({placeholders})
            """,
            mine_ids_list,
        )
        return
    if stage_name == "detail_parse":
        conn.execute(
            f"""
            UPDATE mines
            SET detail_parse_completed_at = NULL
            WHERE mine_id IN ({placeholders})
            """,
            mine_ids_list,
        )
        return
    raise ValueError(f"Unsupported stage name: {stage_name}")


def count_scraped_ids(conn: duckdb.DuckDBPyConnection) -> tuple[int, int]:
    row = conn.execute("""
        SELECT COUNT(*) AS total,
               COUNT(COALESCE(detail_exports_completed_at, detail_scraped_at)) AS scraped
        FROM mines
    """).fetchone()
    return row[0], row[1]


def get_screener_state(conn: duckdb.DuckDBPyConnection, screener_key: str) -> dict | None:
    row = conn.execute("""
        SELECT screener_key, total_pages, last_page_done, started_at, completed_at
        FROM screener_state
        WHERE screener_key = ?
    """, [screener_key]).fetchone()
    if row is None:
        return None
    return {
        "screener_key": row[0],
        "total_pages": row[1],
        "last_page_done": row[2],
        "started_at": row[3],
        "completed_at": row[4],
    }


def init_screener_state(conn: duckdb.DuckDBPyConnection, screener_key: str, total_pages: int) -> None:
    now = datetime.now(timezone.utc)
    conn.execute("""
        INSERT INTO screener_state
            (screener_key, total_pages, last_page_done, started_at, completed_at)
        VALUES (?, ?, 0, ?, NULL)
        ON CONFLICT (screener_key) DO UPDATE SET
            total_pages    = excluded.total_pages,
            last_page_done = 0,
            started_at     = excluded.started_at,
            completed_at   = NULL
    """, [screener_key, total_pages, now])


def save_screener_progress(conn: duckdb.DuckDBPyConnection, screener_key: str, last_page_done: int) -> None:
    conn.execute("""
        UPDATE screener_state
        SET last_page_done = ?
        WHERE screener_key = ?
    """, [last_page_done, screener_key])


def mark_screener_complete(conn: duckdb.DuckDBPyConnection, screener_key: str) -> None:
    conn.execute("""
        UPDATE screener_state
        SET completed_at = ?
        WHERE screener_key = ?
    """, [datetime.now(timezone.utc), screener_key])


def log_error(conn: duckdb.DuckDBPyConnection, mine_id: str, block_name: str, error_msg: str) -> None:
    conn.execute("""
        INSERT INTO scrape_errors (mine_id, block_name, error_msg, occurred_at)
        VALUES (?, ?, ?, ?)
    """, [mine_id, block_name, error_msg, datetime.now(timezone.utc)])
